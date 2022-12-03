from vesc import VESC
from video import VideoStream
from facial_detection import FaceDetector
import time


class SentryMode:
    def __init__(self, target_im_path, port="/dev/ttyACM0", angle=0.3, throttle=0.15, tolerance=0.6):
        self.stream = VideoStream()
        self.vesc = VESC(port)

        self.face_detector = FaceDetector()
        self.face_detector.learn_face(target_im_path)
        self.tolerance = tolerance

        self.target_im_path = target_im_path
        self.port = port
        self.angle = angle
        self.throttle = throttle

    def step(self, sleep=0.1):

        self.vesc.run(angle=self.angle, throttle=self.throttle)

        rgb = self.stream.get_rgb()
        if rgb is not None:
            result = self.face_detector.detect_face(rgb, tolerance=self.tolerance)
        else:
            print("no rgb detected")
            result = None

        time.sleep(sleep)
        return result


class KillerMode:
    def __init__(self, sentry_obj, throttle=0.4, steering_scale=0.5, tolerance=0.6):
        self.stream = sentry_obj.stream
        self.vesc = sentry_obj.vesc
        self.face_detector = sentry_obj.face_detector
        self.tolerance = tolerance
        self.throttle = throttle
        self.steering_scale=steering_scale

    def step(self, sleep=0.1):
        control_angle = 0
        control_throttle = 0

        rgb = self.stream.get_rgb()
        result = []
        if rgb is not None:
            result = self.face_detector.detect_face(rgb, tolerance=self.tolerance)

            if result:
                print("face detected!")
                control_angle = self.compute_control(rgb, result) * self.steering_scale
                control_throttle = self.throttle

        self.vesc.run(angle=control_angle, throttle=control_throttle)

        time.sleep(sleep)
        return result

    def compute_control(self, im, bb):
        sizex, sizey, _ = im.shape
        top, right, bottom, left = bb[0]

        midpoint = (right + left) / 2

        midpoint -= (sizey // 2)
        xdist = midpoint / (sizey // 2)

        return xdist


if __name__ == '__main__':
    SENTRY_THROTTLE = 0.15
    SENTRY_ANGLE = 0.3
    SENTRY_TOLERANCE = 0.8  # The default is 0.6, increase to make more permissive and decrease to make more strict
    SENTRY_SLEEP = 0.25

    KILLER_THROTTLE = 0.4
    KILLER_SCALE = 0.3
    KILLER_TOLERANCE = 1.0
    KILLER_SLEEP = 0
    TARGET_FACE_PATH = "alex.jpg"  # Put the path to the image of the target face

    SERIAL_PORT = "/dev/ttyACM0"

    sentry = SentryMode(
        TARGET_FACE_PATH,
        port=SERIAL_PORT,
        angle=SENTRY_ANGLE,
        throttle=SENTRY_THROTTLE,
        tolerance=SENTRY_TOLERANCE
    )

    killer = KillerMode(
        sentry,
        throttle=KILLER_THROTTLE,
        steering_scale=KILLER_SCALE,
        tolerance=KILLER_TOLERANCE
    )


    print("The hunt is afoot")
    while True:
        result = sentry.step(SENTRY_SLEEP)
        if result:
            break

    print("Gotcha bitch!")
    print("you can run, but you can't hide, bitch")
    while True:
        result = killer.step(KILLER_SLEEP)
        if not result:
            break

    print("I guess you can hide")


