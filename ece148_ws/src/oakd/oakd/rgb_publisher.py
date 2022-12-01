import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import depthai
import numpy as np


class RGBPublisher(Node):

    def __init__(self):
        super().__init__("rgb")
        self.publisher = self.create_publisher(Image, "rgb", 10)

        timer_period = 0.1  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

        pipeline = depthai.Pipeline()
        cam_rgb = pipeline.create(depthai.node.ColorCamera)
        cam_rgb.setPreviewSize(300, 300)
        cam_rgb.setInterleaved(False)

        xout_rgb = pipeline.create(depthai.node.XLinkOut)
        xout_rgb.setStreamName("rgb")
        cam_rgb.preview.link(xout_rgb.input)

        self.pipeline = pipeline
        self.cam_rgb = cam_rgb

        self.device = depthai.Device(pipeline).__enter__()
        self.q_rgb = self.device.getOutputQueue("rgb", maxSize=1)
        self.bridge = CvBridge()

    def timer_callback(self):

        in_rgb = self.q_rgb.tryGet()

        if in_rgb is not None:
            frame = in_rgb.getCvFrame()
            msg = self.bridge.cv2_to_imgmsg(np.array(frame), "bgr8")

            self.publisher.publish(msg)
            self.get_logger().info('Publishing RGB')
            self.i += 1


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





