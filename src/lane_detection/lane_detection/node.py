import rclpy
import lane_detection

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from .detect_lanes import detect

import cv2
from cv_bridge import CvBridge


class LaneDetection(Node):

    def __init__(self):
        super().__init__("lane_detection")
        self.left_camera_subscription = self.create_subscription(
            Image, "/camera1/image_raw", self.listener_callback, 10
        )
        # self.right_camera_subscription = self.create_subscription(
        #     Image, "/short_1_camera/image_raw", self.listener_callback, 10
        # )

    def listener_callback(self, msg):
        current_frame = CvBridge().imgmsg_to_cv2(msg)
        detect(current_frame)


def main(args=None):
    rclpy.init(args=args)

    node = LaneDetection()

    rclpy.spin(node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
