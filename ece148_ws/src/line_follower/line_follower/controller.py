"""
Subscribes to the rgb channel,
computes control and publishes to the control channel
"""


import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import depthai
import numpy as np
from line_follower3 import Controller as Ctrl


class Controller(Node):

    def __init__(self):
        super().__init__("rgb")
        self.subscriber = self.create_subscription(Image, "rgb", 10)
        self.publisher = self.create_publisher(Twist, "control", self.listener_callback, 10)
        timer_period = 0.1  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0
        self.controller = Ctrl()
        self.bridge = CvBridge()

        self.control_msg = Twist()
        self.control_msg.linear.x = 0
        self.control_msg.angular.x = 0

    def timer_callback(self):
        self.publisher.publish(self.control_msg)
        self.i += 1

    def listener_callback(self, msg):
        img = self.bridge.imgmsg_to_cv2(np.array(frame), "rgb8")
        steering = self.controller.get_control(img)
        throttle = 0.2
        self.control_msg.linear.x = throttle
        self.control_msg.angular.x = steering


def main(args=None):
    rclpy.init(args=args)

    publisher = RGBPublisher()

    rclpy.spin(publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()






