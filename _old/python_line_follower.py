import cv2
# import depthai as dai
import time
# import pyvesc




class Vesc:
    """
    Takes an input and sends it to the vesc.
    """
    pass


class OAKD:
    def __init__(self, image_w=160, image_h=120, image_d=3, framerate=20, camera_index=0):

        # initialize variable used to indicate
        # if the thread should be stopped
        self.frame = None
        self.image_d = image_d
        self.image_w = image_w
        self.image_h = image_h
        self.framerate = framerate

        self.init_camera(image_w, image_h, image_d)
        self.on = True

    def init_camera(self, image_w, image_h, image_d, camera_index=0):


        self.resolution = (image_w, image_h)
        self.pipeline = dai.Pipeline()

        # Define a source - color camera
        self.camRgb = self.pipeline.create(dai.node.ColorCamera)
        self.camRgb.setPreviewSize(self.image_w, self.image_h)
        self.camRgb.setInterleaved(False)

        self.videoEnc = self.pipeline.create(dai.node.VideoEncoder)

        self.camControlIn = self.pipeline.create(dai.node.XLinkIn)
        self.camControlIn.setStreamName('camControl')
        self.camControlIn.out.link(self.camRgb.inputControl)

        # # Create output
        self.xoutRgb = self.pipeline.createXLinkOut()
        self.xoutRgb.setStreamName("rgb")

        self.xoutRgb_video = self.pipeline.createXLinkOut()
        self.xoutRgb_video.setStreamName("video")

        self.camRgb.preview.link(self.xoutRgb.input)
        self.videoEnc.setDefaultProfilePreset(30, dai.VideoEncoderProperties.Profile.H265_MAIN)
        self.camRgb.video.link(self.videoEnc.input)
        self.videoEnc.bitstream.link(self.xoutRgb_video.input)

        self.start_time = time.time()
        self.device = dai.Device(self.pipeline)
        self.start_time = time.time()
        while True:
            controlQueue = self.device.getInputQueue('camControl')
            ctrl = dai.CameraControl()
            self.device.startPipeline()
            ctrl.setAutoExposureEnable()
            controlQueue.send(ctrl)
            self.end_time = time.time()
            if ((self.end_time - self.start_time) > 6):
                self.qRgb = self.device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
                self.qRgb_video = self.device.getOutputQueue(name="video", maxSize=4, blocking=True)
                self.videofile = open('output_video.h265', 'wb')

                print("Exposure set for the current environment...OAKD STARTED........")
                break

    def run(self, qRgb, qRgb_video):
        inRgb = qRgb.get()
        self.frame = inRgb.getCvFrame()
        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        inRgb_video = qRgb_video.get()
        inRgb_video.getData().tofile(self.videofile)
        # print("Image_Captured")
        return self.frame

    def update(self):
        from datetime import datetime, timedelta
        while self.on:
            start = datetime.now()
            self.run(self.qRgb, self.qRgb_video)
            stop = datetime.now()
            s = 1 / self.framerate - (stop - start).total_seconds()
            if s > 0:
                time.sleep(s)

    def run_threaded(self):
        return self.frame



class Controller:
    """
    Takes an input image and computes the throttle and
    """

    def process_image(self, rgb):
        """

        :param rgb: np.array, HxWx3
        :return: (throttle, angle)

            throttle is [0, 1], angle is [0, 1]
        """

