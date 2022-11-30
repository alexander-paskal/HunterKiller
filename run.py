from VESC import VESC
from line_follower import Controller
import depthai
import cv2
import matplotlib.pyplot as plt

if __name__ == '__main__':

    period = 50

    pipeline = depthai.Pipeline()
    cam_rgb = pipeline.create(depthai.node.ColorCamera)
    cam_rgb.setPreviewSize(300, 300)
    cam_rgb.setInterleaved(False)

    xout_rgb = pipeline.create(depthai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")
    cam_rgb.preview.link(xout_rgb.input)

    ##fct
    with depthai.Device(pipeline) as device:
        q_rgb = device.getOutputQueue("rgb", maxSize=1)

        frame = None
        detections = []

        prev = []
        controller = Controller()
        vesc = VESC("/dev/ttyACM0")
        while True:
            in_rgb = q_rgb.tryGet()

            if in_rgb is None:
                continue

            frame = in_rgb.getCvFrame()

            steering = controller.get_control(frame)
            # prev.append(steering)
            # plt.clf()
            # plt.imshow(controller.colormask(frame))
            # plt.draw()
            # plt.pause(1/30)
            # smooth_steering = sma(prev, period)
            # steering = max([min([0.3, smooth_steering]), -0.3])
            if steering is not None:
                vesc.run(steering, .2)
                print(steering)