import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

COLORMIN = 100
COLORMAX = 150
H = 300
W = 300
ROW_THRESHOLD = 2000
COL_THRESHOLD = 2000
ANGLE_WEIGHT = 1
OFFSET_WEIGHT = 0
PERIOD = 10
SCALE_STEERING = 1.2


class Controller:
    def get_control(self, mask):
        # mask = self.colormask(im)

        rows = [
            (mask[:60], 0),
            (mask[60:120], 60),
            (mask[120:180], 120),
            (mask[180:240], 180),
            (mask[240:], 240)
        ][::-1]

        cols = [
            (mask[150:, :60], 0),
            (mask[150:, 60:120], 60),
            (mask[150:, 120:180], 120),
            (mask[150:, 180:240], 180),
            (mask[150:, 240:], 240)
        ]

        angle = self.compute_angle(rows, cols)
        steering = self.angle2steering(angle)
        return steering
        # return angle, steering


    def angle2steering(self, angle):
        if np.isnan(angle):
            return None
        angle = np.degrees(angle)

        angle = 180 - angle



        angle /= 18
        angle = round(angle, 2)
        angle /= 10

        if angle > 0.5:
            angle *= SCALE_STEERING
        elif angle < 0.5:
            angle /= SCALE_STEERING

        return angle



    def compute_angle(self, rows, cols):

        rows = [row for row in rows if np.sum(row[0].flatten()) > ROW_THRESHOLD]
        if rows:
            back_cm = self.center_mass(rows[0][0], yoff=rows[0][1])
            front_cm = self.center_mass(rows[-1][0], yoff=rows[-1][1])

            vector = front_cm - back_cm
            vector = vector / W
            # return vector
            cos = np.dot(vector, np.array([1, 0])) / np.linalg.norm(vector)
            angle = np.arccos(cos)
            return angle

        return np.pi/2


        #
        #     if not np.isnan(cos):
        #         angle_val = cos
        #
        # offset_val = 0
        # if np.sum(np.concatenate([col[0] for col in cols]).flatten()) > COL_THRESHOLD:
        #     weights = []
        #     offsets = []
        #     for col, offset in cols:
        #         weight = np.sum(col.flatten()) / (150*60)
        #         weights.append(weight)
        #         offsets.append(offset + 30)
        #
        #     # weights = np.array(weights) / np.linalg.norm(weights)
        #     center = np.dot(weights, offsets) - 150
        #     offset_val = center / 300
        #
        # # angle = angle_val * ANGLE_WEIGHT + offset_val * OFFSET_WEIGHT
        #
        # return angle_val

    def colormask(self, im):
        colormin, colormask = COLORMIN, COLORMAX
        # im = cv2.blur(im, (3, 3), 0)
        shape = im.shape
        im = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
        # return im[:, :, 2]
        im = im.reshape((-1, 3))

        mask = np.logical_and(
            im[:, 0] > colormin,
            im[:, 0] < colormask
        )

        mask = mask.reshape(shape[:2])
        return mask

    def center_mass(self, mask, xoff=0, yoff=0):
        H, W = mask.shape
        X, Y = np.meshgrid(
            np.arange(W), np.arange(H)
        )
        X = X[mask]
        Y = Y[mask]

        xcm = int(np.mean(X.flatten())) + xoff
        ycm = int(np.mean(Y.flatten())) + yoff

        return np.array([xcm, -ycm])


def sma(l, period):
    total = np.sum(l[-period:])
    return total/period


def roi_from_lines(lines):
    # lines = lines.squeeze()
    roi_points = []

    for line in lines:
        line = line.squeeze()
        roi_points.append(line[:2])
        roi_points.append(line[2:])

    return np.array(roi_points, dtype=np.int32)

def show_image(name, img):  # function for displaying the image
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def find_canny(img, thresh_low, thresh_high):  # function for implementing the canny
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # show_image('gray',img_gray)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    # show_image('blur',img_blur)
    img_canny = cv2.Canny(img_blur, thresh_low, thresh_high)
    # show_image('Canny', img_canny)
    return img_canny


def region_of_interest(image, bounds):  # function for extracting region of interest
    # bounds in (x,y) format

    bounds = bounds.reshape((1, -1, 2))
    # bounds = np.array([[[0,image.shape[0]],[0,image.shape[0]/2],[900,image.shape[0]/2],[900,image.shape[0]]]],dtype=np.int32)
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, bounds, [255, 255, 255])
    # show_image('inputmask',mask)
    masked_image = cv2.bitwise_and(image, mask)
    # show_image('mask', masked_image)
    return masked_image, mask


def draw_lines(img, lines):  # function for drawing lines on black mask
    mask_lines = np.zeros_like(img)
    for points in lines:
        x1, y1, x2, y2 = points[0]
        cv2.line(mask_lines, (x1, y1), (x2, y2), [0, 0, 255], 2)

    return mask_lines


def get_coordinates(img, line_parameters):  # functions for getting final coordinates
    slope = line_parameters[0]
    intercept = line_parameters[1]
    y1 = 300
    y2 = 120
    # y1=img.shape[0]
    # y2 = 0.6*img.shape[0]
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return [x1, int(y1), x2, int(y2)]


if __name__ == '__main__':
    i = 29130
    controller = Controller()
    fig, axs = plt.subplots(1, 2)

    angles = []
    smas = []
    while True:
        fname = os.path.join("traj", f"{i}.jpg")

        bgr = cv2.imread(fname)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        lane_canny = find_canny(bgr, 100, 200)
        bounds = np.array(
            [[[0, rgb.shape[0]], [0, rgb.shape[0] / 2], [300, rgb.shape[0] / 2], [300, rgb.shape[0]]]],
            dtype=np.int32)
        # lane_roi = region_of_interest(lane_canny, bounds)


        lane_roi = lane_canny
        lane_lines = cv2.HoughLinesP(lane_roi, 1, np.pi / 180, 50, 40, 5)

        im = rgb
        mask = np.zeros((300, 300))
        if lane_lines is not None:
            bounds = roi_from_lines(lane_lines)
            im, mask = region_of_interest(im, bounds)
            mask = np.array(cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY), dtype=bool)


        axs[0].clear()
        axs[0].imshow(im)

        angle, steering = controller.get_control(mask)
        print(np.degrees(angle), steering)
        # if not isinstance(vector, int):
        #     axs[1].clear()
        #     axs[1].quiver(0, 0, *vector, scale=3)


        plt.draw()
        plt.pause(1 / 30)
        i += 30

