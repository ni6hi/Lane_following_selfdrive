from rclpy.node import Node
import rclpy
from sensor_msgs.msg import Image
import cv2
import numpy as np
import math
from cv_bridge import CvBridge
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String

class IPM_PROCESSOR(Node):

    def __init__(self):
        super().__init__('pubsub')
        self.bridge = CvBridge()

        #creating publishers and subscribers
        self.ipm_sub = self.create_subscription(String, '/igvc/xypoints', self.ipm_callback, 10)
        self.ipm_sub

    def quaternion_to_euler(self, x, y, z, w):

        import math
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        X = math.degrees(math.atan2(t0, t1))

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        Y = math.degrees(math.asin(t2))

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        Z = math.degrees(math.atan2(t3, t4))

        return X, Y, Z

    def getPerpCoord(self, lane, aX, aY, bX, bY, length):
        vX = bX-aX
        vY = bY-aY
        mag = math.sqrt(vX*vX + vY*vY)
        vX = vX / mag
        vY = vY / mag
        temp = vX
        vX = 0-vY
        vY = temp
        cX = bX + 2*lane*(vX * length)
        cY = bY + 2*lane*(vY * length)
        dX = aX + lane*(vX * length)
        dY = aY + lane*(vY * length)
        return int(cX), int(cY), int(dX), int(dY)

    def ipm_callback(self, data):
        points = [data.data[i:i+3] for i in range(0, len(data.data), 3)]
        print(points[0], points[1])

def main(args=None):
    rclpy.init(args=args)

    ipmnode = IPM_PROCESSOR()   
    rclpy.spin(ipmnode)
    ipmnode.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
