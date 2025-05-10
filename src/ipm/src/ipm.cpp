#include <functional>
#include <math.h>
#include <memory>
#include <string>
#include <Eigen/Dense>
#include "cv_bridge/cv_bridge.h"
#include <pcl/point_types.h>
#include "tf2/exceptions.h"
#include "tf2_ros/transform_listener.h"
#include "tf2_ros/buffer.h"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/pcd_io.h>
#include "std_msgs/msg/string.hpp"
#include "std_msgs/msg/header.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/msg/point_field.hpp"
typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;

using namespace std;
using namespace Eigen;
using std::placeholders::_1;

class IPM : public rclcpp::Node
{
  public:
    IPM()
    : Node("ipm")
    {
       this->set_parameter(rclcpp::Parameter("use_sim_time", true));
       left_caminfo_subscription = this->create_subscription<sensor_msgs::msg::CameraInfo>(
			   "/camera1/camera_info", 10, std::bind(&IPM::left_info_callback, this, _1));

       left_img_subscription = this->create_subscription<sensor_msgs::msg::Image>(
			   "/model_lanes", 10, std::bind(&IPM::left_img_callback, this, _1));

       right_caminfo_subscription = this->create_subscription<sensor_msgs::msg::CameraInfo>(
			   "/short_1_camera/camera_info", 10, std::bind(&IPM::right_info_callback, this, _1));

       right_img_subscription = this->create_subscription<sensor_msgs::msg::Image>(
			   "/model_lanes2", 10, std::bind(&IPM::right_img_callback, this, _1));

       publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/igvc/ipm", 10);

       tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
       tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

	   while(true){
		    std::string toFrameRel = "base_link";
			try 
			{
				t1 = tf_buffer_->lookupTransform(
						toFrameRel, "camera_short_link",
						tf2::TimePointZero);
				t2 = tf_buffer_->lookupTransform(
						toFrameRel, "camera_link",
						tf2::TimePointZero);
	
			RCLCPP_INFO(this->get_logger(), "transform received! starting ipm...");
				break;
			} 
			catch (const tf2::TransformException & ex) 
			{
			RCLCPP_INFO(
					this->get_logger(), "ERROR: %s, transform from %s to %s not available",
					ex.what(), toFrameRel.c_str(), "camera frame");
			}
	   }

    }

  private:
    void right_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
    {
        this->right_camera_info= *msg;
    }
    void left_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
    {
        this->left_camera_info= *msg;
    }
    void right_img_callback(const sensor_msgs::msg::Image::SharedPtr msg){
		process_img(msg, "camera_short_link", this->right_camera_info);
		return;
    }

    void left_img_callback(const sensor_msgs::msg::Image::SharedPtr msg){
		process_img(msg, "camera_link", this->left_camera_info);
    }
    void process_img(const sensor_msgs::msg::Image::SharedPtr msg, std::string &&frame, sensor_msgs::msg::CameraInfo &camera_info)
    {
		// processing recieved image
		sensor_msgs::msg::PointCloud2 pub_pointcloud;
		unique_ptr<PointCloud> cloud_msg  = std::make_unique<PointCloud>();
		cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::RGB8);
		cv::Mat gray_image;
		cv::cvtColor(cv_ptr->image, gray_image, cv::COLOR_RGB2GRAY);
		cv::inRange(gray_image, cv::Scalar(245), cv::Scalar(255), gray_image);
		cv::Mat nonZeroCoordinates;
		cv::findNonZero(gray_image, nonZeroCoordinates);

		/*
		 * ipm code: refer to https://thomasfermi.github.io/Algorithms-for-Automated-Driving/LaneDetection/InversePerspectiveMapping.html
		 */

		auto t = (frame == "camera_link")?t2:t1;

		// There are two different axes, one followed 
		// by the base link and the other used in the code
		// for ipm calculation. 
		// assume xyz to be base link and x'y'z' to 
		// be ipm axes. then the relation holds:
		//
		// x == -z'
		// y == x'
		// z == -y'
		//
		// these quaternions are in the base_link axes 
		double q0 = t.transform.rotation.w;
		double q1 = t.transform.rotation.x;
		double q2 = t.transform.rotation.y;
		double q3 = t.transform.rotation.z;

		Eigen::Matrix<double, 3, 3> k;
		k(0, 0) = 2 * (q0 * q0 + q1 * q1) - 1;
		k(0, 1) = 2 * (q1 * q2 - q0 * q3);
		k(0, 2) = 2 * (q1 * q3 + q0 * q2);
		k(1, 0) = 2 * (q1 * q2 + q0 * q3);
		k(1, 1) = 2 * (q0 * q0 + q2 * q2) - 1;
		k(1, 2) = 2 * (q2 * q3 - q0 * q1);
		k(2, 0) = 2 * (q1 * q3 - q0 * q2);
		k(2, 1) = 2 * (q2 * q3 + q0 * q1);
		k(2, 2) = 2 * (q0 * q0 + q3 * q3) - 1;

		auto k_inv = k.inverse();
		// normal vector to road in base link axes
		Eigen::Matrix<double, 3, 1> nor;
		nor(0,0) = 0.0f;
		nor(1,0) = 0.0f;
		nor(2,0) = 1.0f;

		Eigen::Matrix<double, 1, 3> nT_base_link_axes = (k_inv * nor).transpose();
									
		// transformed normal vector( ncT ) in ipm axes
		Eigen::Matrix<double, 1, 3> nT;
		// x' == y
		// y' == -z
		// z' ==  -x
		nT[0] = nT_base_link_axes[1];
		nT[1] = -nT_base_link_axes[2];
		nT[2] = -nT_base_link_axes[0];

		cerr << nT << endl;

		// no of points to map
		cv::Size s = nonZeroCoordinates.size();
		unsigned int cols = s.height;
		std::cout << "cols: " << cols << '\n';
		assert(cols > 0);

		// K matrix is camera's intrinsic parameters 
		auto caminfo = camera_info.k;
		Eigen::Map<Matrix<double,3,3,RowMajor> > camera_params(caminfo.data());
		auto inv_caminfo = camera_params.inverse();

		// uv_hom contains coordinates in the image
		// why is this named _hom?
		Eigen::Matrix<double, 3, Dynamic> uv_hom(3, cols);
		for(int i=0;i < cols; ++i){
			int x = nonZeroCoordinates.at<cv::Point>(i).x;
			int y = nonZeroCoordinates.at<cv::Point>(i).y;
			uv_hom(0, i) = x;
			uv_hom(1, i) = y;
			uv_hom(2, i) = 1;
		}

		//kin_uv  = K^(-1) * uv
		Eigen::Matrix<double, 3, Dynamic> kin_uv(3, cols);
		kin_uv = inv_caminfo * uv_hom;

		// denominator to divide by when mapping points 
		Eigen::Matrix<double, 1, Dynamic> denom(1, cols);
		denom = nT * kin_uv;
		double camera_height = t.transform.translation.z;
		for(int i=0; i < cols; ++i){
			// vec is points in road frame
			pcl::PointXYZ vec;
			
			// mapped points in base link axes 
			Eigen::Matrix<double, 3, 1> camera_frame_point;
			// x == -z'
			// y == x'
			// z == -y'
			camera_frame_point[0] =  -camera_height * kin_uv(2, i) / denom(0,i);
			camera_frame_point[1] = camera_height * kin_uv(0, i) / denom(0,i);
			camera_frame_point[2] = -camera_height * kin_uv(1, i) / denom(0,i);

			// transforming from camera frame to base link frame
			auto temp = k * camera_frame_point;
			vec.x = temp[0];
			vec.y = temp[1];
			// there is a z translation between camera frame and 
			// base link
			vec.z = temp[2] - camera_height;

			cloud_msg->points.push_back(vec);
		}

		cloud_msg->height   = 1;
		cloud_msg->width    = cloud_msg->points.size();
		cloud_msg->is_dense = false;
		pcl::toROSMsg(*cloud_msg, pub_pointcloud);
		pub_pointcloud.header.frame_id = "base_link";//frame;
		pub_pointcloud.header.stamp = this->now();
		publisher_->publish(pub_pointcloud);

		cloud_msg->points.clear();
    }

    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr right_caminfo_subscription;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr right_img_subscription;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr left_caminfo_subscription;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr left_img_subscription;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;
	std::shared_ptr<tf2_ros::TransformListener> tf_listener_{nullptr};
	std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
	geometry_msgs::msg::TransformStamped t1, t2;
    sensor_msgs::msg::CameraInfo right_camera_info;
    sensor_msgs::msg::CameraInfo left_camera_info;
};

int main(int argc, char * argv[])
{
	rclcpp::init(argc, argv);
	rclcpp::spin(std::make_shared<IPM>());
	rclcpp::shutdown();
	return 0;
}

