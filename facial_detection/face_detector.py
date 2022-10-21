import numpy as np


class FaceDetector:
    def __init__(self):
        pass

    def learn_face(self, im):
        """
        Given an image of a face, update the FaceDetector internal model
        to be able to detect instances of that face in images
        :param im: np.ndarray with dimensions HxWx3, color image with RGB channels (as opposed to BGR)
        :return:
        """

        raise NotImplementedError

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
        raise NotImplementedError