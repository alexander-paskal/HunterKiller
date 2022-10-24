import numpy as np
import imutils
import cv2
import os
import face_recognition
from base64 import b64decode
'''from google.colab.patches import cv2_imshow'''

class FaceDetector:
    def __init__(self):
        self.target_face_encoding = None
        self.target_face_name = None

    def learn_face(self, im):
        """
        Given an image of a face, update the FaceDetector internal model
        to be able to detect instances of that face in images
        :param im: np.ndarray with dimensions HxWx3, color image with RGB channels (as opposed to BGR)
        :return:
        """
        
        # Load a sample picture and learn how to recognize it.
        sample1_image = face_recognition.load_image_file(os.path.join(im))
        sample1_face_encoding = face_recognition.face_encodings(sample1_image)[0]

        # Create arrays of known face encodings and their names
        self.target_face_encoding = [
            sample1_face_encoding,
        ]
        self.target_face_name = [
            "Jon Snow",
        ]

    def detect_face(self, im):
        """
        Given an image, determine where, if anywhere the face is present in the image
        and return the bounding box coordinates of the most likely face candidate

        If a candidate is found, return only one array of

            [upperx, uppery, lowerx, lowery]

        where the values correspond to the pixels of the image on which the upper left
        and lower right corners of the bounding box lie

        If no face is found, return None
        :param im: np.ndarray with dimensions HxWx3, color image with RGB channels (as opposed to BGR)
        :return: Optional[np.ndarray], the 4-dimensional vector of bounding box pixel coordinates described above

        """
        # Load video
        videoFile = "facial_detection\jon_snow_video.mp4"
        cap = cv2.VideoCapture(os.path.join(videoFile))

        # Initialize some variables
        face_locations = []
        face_encodings = []
        face_names = []
        process_this_frame = True

        while True:
            # Grab a single frame of video
            ret, frame = cap.read()

            # Only process every other frame of video to save time
            if process_this_frame:
                # Resize frame of video to 1/4 size for faster face recognition processing
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)

                # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
                rgb_small_frame = small_frame[:, :, ::-1]
                
                # Find all the faces and face encodings in the current frame of video
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                face_names = []
                for face_encoding in face_encodings:
                    # See if the face is a match for the known face(s)
                    matches = face_recognition.compare_faces(self.target_face_encoding, face_encoding)
                    name = "Unknown"

                    # # If a match was found in known_face_encodings, just use the first one.
                    # if True in matches:
                    #     first_match_index = matches.index(True)
                    #     name = known_face_names[first_match_index]

                    # Or instead, use the known face with the smallest distance to the new face
                    face_distances = face_recognition.face_distance(self.target_face_encoding, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = self.target_face_name[best_match_index]

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

    def read_im(self, im_path):
        pass

    def save_as_video(self, frames, fpath):
        pass


face_detector = FaceDetector()
'''sample image'''
face_file = r"facial_detection\jon_snow_face.jpg"
# im=cv2.imread(image)
input_video= r"facial_detection\jon_snow_video.mp4"
# outputFile = r"HunterKiller\VtoF"
# face_detector.video_to_frames(inputFile, outputFile)
# face_detector.learn_face(im)

# cap= cv2.VideoCapture(inputFile)
# face_locations = []

# while True:
#     # Grab a single frame of video
#     ret, frame = cap.read()
#     # Convert the image from BGR color (which OpenCV uses) to RGB   
#     # color (which face_recognition uses)
#     rgb_frame = frame[:, :, ::-1]
#     # Find all the faces in the current frame of video
#     face_locations = face_recognition.face_locations(rgb_frame)
#     for top, right, bottom, left in face_locations:
#         # Draw a box around the face
#         cv2.rectangle(frame, (left, top), (right, bottom), (0, 0,  
#         255), 2)
#     # Display the resulting image
#     cv2.imshow('Video', frame)
    

#     # Wait for Enter key to stop
#     if cv2.waitKey(25) == 13:
#         break

face_detector.learn_face(face_file)
face_detector.detect_face(input_video)
print("It worked!")