import matplotlib.pyplot as plt
from vesc import VESC
from video import VideoStream
from facial_detection import FaceDetector
import cv2


STEERING_SCALE = 0.3
THROTTLE = 0
TOLERANCE = 0.8  # The default is 0.6, increase to make more permissive and decrease to make more strict
TARGET_FACE_PATH = "alex.jpg"  # Put the path to the image of the target face


class PseudoVesc():
    def run(self, angle, throttle):
        color = "blue"

        if angle < 0:
            color = "red"

        data = [0, 0, angle, 0, 0]
        plt.clf()
        plt.barh(range(5), data, color=color)
        plt.xlim(-1, 1)
        plt.axvline(x=0, color='k')
        plt.yticks([])
        plt.draw()
        plt.pause(0.1)


def plot_image(im, bb):

    if bb:
        top, right, bottom, left = bb[0]
        cv2.rectangle(im, (right, bottom), (left, top), (255, 0, 0), 2)

    cv2.imshow("preview", im)


def compute_control(im, bb):
    sizex, sizey, _ = im.shape
    top, right, bottom, left = bb[0]

    midpoint = (right + left) / 2

    midpoint -= (sizey // 2)
    xdist = midpoint / (sizey // 2)

    return xdist


if __name__ == '__main__':
    stream = VideoStream()
    vesc = VESC("/dev/ttyACM0")
    # vesc = PseudoVesc()

    face_detector = FaceDetector()

    face_detector.learn_face(TARGET_FACE_PATH)

    while True:
        control_angle = 0
        control_throttle = 0
        rgb = stream.get_rgb()

        if rgb is not None:
            print("RGB image detected")
            result = face_detector.detect_face(rgb, tolerance=TOLERANCE)
            print("result")
            if result:
                print("face detected!")
                control_angle = compute_control(rgb, result) * STEERING_SCALE
                throttle = THROTTLE
            else:
                print("no face detected")
                control_angle = 0
                throttle = 0
            plot_image(rgb, result)

        vesc.run(throttle=control_throttle, angle=control_angle)

        if cv2.waitKey(1) == ord('q'):
            break