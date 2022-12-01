import depthai as dai


class VideoStream:
    def __init__(self):
        pipeline = dai.Pipeline()
        cam_rgb = pipeline.create(dai.node.ColorCamera)
        cam_rgb.setPreviewSize(300, 300)
        cam_rgb.setInterleaved(False)

        xout_rgb = pipeline.create(dai.node.XLinkOut)
        xout_rgb.setStreamName("rgb")
        cam_rgb.preview.link(xout_rgb.input)

        # depth
        # Closer-in minimum depth, disparity range is doubled (from 95 to 190):
        extended_disparity = False
        # Better accuracy for longer distance, fractional disparity 32-levels:
        subpixel = False
        # Better handling for occlusions:
        lr_check = True

        # Define sources and outputs
        monoLeft = pipeline.create(dai.node.MonoCamera)
        monoRight = pipeline.create(dai.node.MonoCamera)
        depth = pipeline.create(dai.node.StereoDepth)
        xout = pipeline.create(dai.node.XLinkOut)

        xout.setStreamName("disparity")

        # Properties
        monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
        monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

        # Create a node that will produce the depth map (using disparity output as it's easier to visualize depth this way)
        depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        # Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7 (default)
        depth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
        depth.setLeftRightCheck(lr_check)
        depth.setExtendedDisparity(extended_disparity)
        depth.setSubpixel(subpixel)

        # Linking
        monoLeft.out.link(depth.left)
        monoRight.out.link(depth.right)
        depth.disparity.link(xout.input)

        self.pipeline = pipeline

        self.device = dai.Device(pipeline)
        self.device.__enter__()

        self.q_rgb = self.device.getOutputQueue("rgb", maxSize=1, blocking=False)
        self.q_depth = self.device.getOutputQueue(name="disparity", maxSize=1, blocking=False)

    def get_rgb(self):
        in_rgb = self.q_rgb.tryGet()
        if in_rgb is not None:
            rgb_frame = in_rgb.getCvFrame()
            return rgb_frame

        return None

    def get_depth(self):
        in_disparity = self.q_depth.get()  # blocking call, will wait until a new data has arrived

        if in_disparity is not None:
            depth_frame = in_disparity.getCvFrame()
            return depth_frame

        return None
