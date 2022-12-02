from vesc import VESC
from video import VideoStream
from facial_detection import FaceDetector  # to import this, make sure you install the face_recognition package
# to do that, go to HunterKiller/face_recognition-1.3.0 in the terminal and run "pip install -e ."
import time


TARGET_FACE_PATH = "sushovan.png"  # Put the path to the image of the target face


if __name__ == '__main__':
    stream = VideoStream()
    vesc = VESC("/dev/ttyACM0")
    face_detector = FaceDetector()

    face_detector.learn_face(TARGET_FACE_PATH)

    while True:
        rgb = stream.get_rgb()
        detection = False
        if rgb is not None:
            print("RGB image detected")
            result = face_detector.detect_face(rgb)
            if result:
                print("face detected!")
                detection = True
            else:
                print("no face detected")

        if detection:
            vesc.run(angle=0, throttle=0.2)
            time.sleep(0.5)
        else:
            vesc.run(angle=0, throttle=0)

