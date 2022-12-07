import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import depthai as dai


EXTENDED_DISPARITY = False # Closer-in minimum depth, disparity range is doubled (from 95 to 190):
SUBPIXEL = False # Better accuracy for longer distance, fractional disparity 32-levels:
LR_CHECK = True # Better handling for occlusions:


class DepthPublisher(Node):

    def __init__(self):
        super().__init__("depth")
        self.publisher = self.create_publisher(Image, "depth", 10)

        timer_period = 0.1  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

        pipeline = dai.Pipeline()
        monoLeft = pipeline.create(dai.node.MonoCamera)
        monoRight = pipeline.create(dai.node.MonoCamera)
        depth = pipeline.create(dai.node.StereoDepth)
        xout = pipeline.create(dai.node.XLinkOut)

        xout.setStreamName("disparity")

        monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
        monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

        depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        # Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7 (default)
        depth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
        depth.setLeftRightCheck(LR_CHECK)
        depth.setExtendedDisparity(EXTENDED_DISPARITY)
        depth.setSubpixel(SUBPIXEL)

        # Linking
        monoLeft.out.link(depth.left)
        monoRight.out.link(depth.right)
        depth.disparity.link(xout.input)

        self.pipeline = pipeline
        self.depth = depth
        self.mono_left = monoLeft
        self.mono_right = monoRight
        self.xout = xout
        
        self.device = dai.Device(pipeline).__enter__()
        self.q_disparity = self.device.getOutputQueue("disparity", maxSize=1)
        self.bridge = CvBridge()

    def timer_callback(self):

        in_disparity = self.q_disparity.tryGet()

        if in_disparity is not None:
            frame = in_disparity.getCvFrame()
            msg = self.bridge.cv2_to_imgmsg(frame, "8SC1")

            self.publisher.publish(msg)
            self.get_logger().info('Publishing Depth')
            self.i += 1


def main(args=None):
    rclpy.init(args=args)

    publisher = DepthPublisher()

    rclpy.spin(publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
