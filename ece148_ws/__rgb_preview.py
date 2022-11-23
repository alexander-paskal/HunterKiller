import depthai
import cv2
import os
import shutil


if __name__ == '__main__':
    pipeline = depthai.Pipeline()
    cam_rgb = pipeline.create(depthai.node.ColorCamera)
    cam_rgb.setPreviewSize(300, 300)
    cam_rgb.setInterleaved(False)

    xout_rgb = pipeline.create(depthai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")
    cam_rgb.preview.link(xout_rgb.input)

    shutil.rmtree("traj")
    os.mkdir("traj")

    with depthai.Device(pipeline) as device:
        q_rgb = device.getOutputQueue("rgb")

        frame = None
        detections = []

        i = 0
        while True:
            in_rgb = q_rgb.tryGet()

            if in_rgb is not None:
                frame = in_rgb.getCvFrame()

            if frame is not None:
                cv2.imshow("preview", frame)

                if i % 30 == 0:
                    pass
                    cv2.imwrite(f"traj/{i}.jpg", frame)

            if cv2.waitKey(1) == ord('q'):
                break

            if len(os.listdir("traj")) > 1000:
                print("recorded max # images")
                break

            i += 1