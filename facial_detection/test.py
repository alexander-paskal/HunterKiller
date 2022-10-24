import face_recognition
import cv2
import os
import numpy as np

import matplotlib.pyplot as plt

# This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
# other example, but it includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Load a sample picture and learn how to recognize it.

imageFile = "facial_detection\jon_snow_face.jpg"
sample_image = face_recognition.load_image_file(os.path.join(imageFile))
sample_face_encoding = face_recognition.face_encodings(sample_image)[0]


# Create arrays of known face encodings and their names
known_face_encodings = [
    sample_face_encoding,
]
known_face_names = [
    "Jon Snow",
]
videoFile = "facial_detection\jon_snow_video.mp4"
cap = cv2.VideoCapture(os.path.join(videoFile))

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # face_locations = []
    # face_encodings = []
    # face_names = []

    # Grab a single frame of video
    ret, frame = cap.read()

    # Only process every other frame of video to save time
    if process_this_frame:
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25,interpolation=cv2.INTER_AREA)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]
        
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()

# """Get the image, resized the im"""

        # im=imutils.resize(im,width=400)
        # cv2.imshow('image', im)
        # cv2.waitKey()
        # (h,w)=im.shape[:2]
        # print(w,h)
        # blob=cv2.dnn.blobFromImage(cv2.resize(im,(300,300)),1.0,(300,300),(104.0,177.0,123.0))

        # """Sample Image Detector"""
        # #model structure: https://github.com/opencv/opencv/raw/3.4.0/samples/dnn/face_detector/deploy.prototxt
        # #pre-trained weights: https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel
        # prototxt='facial_detection/deploy.prototxt'
        # model='facial_detection/res10_300x300_ssd_iter_140000.caffemodel'
        # net=cv2.dnn.readNetFromCaffe(prototxt,model)

        # net.setInput(blob)
        # detections=net.forward()

        # """"""

        # for i in range(0, detections.shape[2]):

        #     # extract the confidence (i.e., probability) associated with the prediction
        #     confidence = detections[0, 0, i, 2]

        #     # filter out weak detections by ensuring the `confidence` is
        #     # greater than the minimum confidence threshold
        #     if confidence > 0.5:
        #         # compute the (x, y)-coordinates of the bounding box for the object
        #         box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        #         (startX, startY, endX, endY) = box.astype("int")
        #         # draw the bounding box of the face along with the associated probability
        #         text = "{:.2f}%".format(confidence * 100)
        #         y = startY - 10 if startY - 10 > 10 else startY + 10
        #         cv2.rectangle(im, (startX, startY), (endX, endY), (0, 0, 255), 2)
        #         cv2.putText(im, text, (startX, y),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

        # """Show the resulting image"""
        # cv2.imshow("dected_image",im)
        # cv2.waitKey()