
#####imports for the jetson's code

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

######imports for the publisher's code from the website

import rclpy
from rclpy.node import Node
from geometry_msgs.Twist import Twist

COLORMIN = 100
COLORMAX = 150
H = 300
W = 300
ROW_THRESHOLD = 600
COL_THRESHOLD = 600
ANGLE_WEIGHT = 1
OFFSET_WEIGHT = 0
PERIOD = 10
SCALE_STEERING = 1.2

###########publisher code

class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(Twist, 'Controller', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = Twist()
        msg.data = 'Hello World: %d' % self.i
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1


def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = MinimalPublisher()

    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

##############code from the jetson

class Controller:
    def __init__(self):
        self.backup = BackupController()


    def get_control(self, og):
        # mask = self.colormask(im)
        im = process_im(og)

        rows = [
            (im[:60], 0),
            (im[60:120], 60),
            (im[120:180], 120),
            (im[180:240], 180),
            (im[240:], 240)
        ][::-1]

        # inds = [300//15 * i for i in range(15)]
        # arrs = np.split(im, inds)[1:]
        # rows = list(zip(arrs, inds))[1:]

        angle = self.compute_angle(rows)

        if angle is None or np.isnan(angle):
            return self.backup.get_control(og)

        steering = self.angle2steering(angle)
        return steering
        # return angle, steering

    def angle2steering(self, angle):
        if np.isnan(angle):
            return None
        angle = np.degrees(angle)
        if angle is None:
            return None

        angle = 180 - angle
        angle /= 18
        angle = round(angle, 2)
        angle /= 10

        if angle > 0.5:
            angle *= SCALE_STEERING
        elif angle < 0.5:
            angle /= SCALE_STEERING

        return angle

    def compute_angle(self, rows):
        rows = [row for row in rows if np.sum(row[0].flatten()) > ROW_THRESHOLD]
        if rows:
            back_cm = self.center_mass(rows[0][0], yoff=rows[0][1])
            front_cm = self.center_mass(rows[-1][0], yoff=rows[-1][1])

            vector = front_cm - back_cm
            vector = vector / W
            # return vector
            # axs[1].clear()
            # axs[1].quiver(0, 0, *vector)

            cos = np.dot(vector, np.array([1, 0])) / np.linalg.norm(vector)
            angle = np.arccos(cos)
            return angle

        return None

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

class BackupController:
    def get_control(self, im):
        im[260:, :] = 0
        blacks, greens = self.colormask(im)
        # return blacks, greens
        cblack = np.sum(blacks.flatten())
        cgreen = np.sum(greens.flatten())

        if cblack > cgreen:
            return 0.05
        else:
            return 0.95


    def colormask(self, im):
        rgb = cv2.blur(im, (2, 2), 0)
        mask1 = np.sum(rgb, axis=-1) < 100
        mask2 = rgb[:, :, 1] > 0.8 * (rgb[:, :, 0] + rgb[:, :, 2])
        mask3 = np.mean(rgb, axis=-1) < 120

        blackface = np.logical_and(mask1, ~mask2)
        greenmachine = np.logical_and(mask2, mask3)
        # whitepower = np.logical_and(~greenmachine, ~blackface)

        # rgb[whitepower, :] = np.array([0, 0, 0])
        # rgb[greenmachine, :] = np.array([0, 255, 0])
        # rgb[blackface, :] = np.array([0, 0, 255])
        # return mask

        return blackface, greenmachine


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


def roi_from_lines(lines):
    # lines = lines.squeeze()
    roi_points = []

    for line in lines:
        line = line.squeeze()
        roi_points.append(line[:2])
        roi_points.append(line[2:])

    return np.array(roi_points, dtype=np.int32)

def process_im(rgb):
    lane_canny = find_canny(rgb, 100, 200)
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

    return mask

if __name__ == '__main__':
    i = 32280
    controller = Controller()
    backup = BackupController()
    fig, axs = plt.subplots(1, 2)

    angles = []
    smas = []
    while True:
        fname = os.path.join("traj", f"{i}.jpg")

        bgr = cv2.imread(fname)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        axs[0].clear()
        axs[0].imshow(rgb)

        steering = controller.get_control(rgb)
        print(steering)


        #
        # plt.draw()
        # plt.show()
        plt.pause(1/30)
        i += 30
