from open_cv_color_segmentation22 import main
import matplotlib.pyplot as plt


import cv2
import numpy as np
import matplotlib.pyplot as plt

COLORMIN = 20
COLORMAX = 250


def process(im):
    im = cv2.blur(im, (3, 3), 0)
    im = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)

    shape = im.shape
    im = im.reshape((-1, 3))

    mask = np.logical_and(
        im[:, 0] > COLORMIN,
        im[:, 0] < COLORMAX
    )

    im[~mask, 2] = 0
    # im[mask] = 255

    im = im.reshape(shape)# im = im[:,:, 2]

    im = cv2.cvtColor(im, cv2.COLOR_HSV2RGB)
    im = cv2.GaussianBlur(im, (3, 3), 0)
    return im


def colormask(im, band):
    colormin, colormask = band
    # im = cv2.blur(im, (3, 3), 0)
    shape = im.shape
    im = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
    im = im.reshape((-1, 3))

    mask = np.logical_and(
        im[:, 0] > colormin,
        im[:, 0] < colormask
    )

    mask = mask.reshape(shape[:2])
    return mask


def center_mass(mask):
    H, W = mask.shape
    X, Y = np.meshgrid(
        np.arange(W), np.arange(H)
    )
    X = X[mask]
    Y = Y[mask]

    return int(np.mean(Y.flatten())), int(np.mean(X.flatten()))



def extract_edges(im):
    im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    im = cv2.GaussianBlur(im, (5, 5), 0)
    # im = np.uint8(cv2.Sobel(im, cv2.CV_64F, 1, 1, 3))
    canny = cv2.Canny(im, 50, 200)

    return canny


if __name__ == '__main__':
    IM = "/home/alexander/Downloads/road4.jpg"
    im = cv2.imread(IM)

    shape = im.shape
    #
    for i in range(5):
        im = process(im)

    mask = colormask(im, (COLORMIN, COLORMAX))





if __name__ == '__main__':

    path = "data/images/{}_cam_image_array_.jpg"


    for i in range(1000):
        # im = cv2.imread(path.format(i))
        #
        # rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        # rgb[:70, :, :] = 0
        # hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        #
        #
        # mask = np.logical_and(
        #     (hsv[:, :, 0] < 100),
        #     (hsv[:, :, 0] > 75)
        # ).flatten()
        #
        # rgb = rgb.reshape((-1, 3))
        # rgb[mask, :] = 255
        # rgb = rgb.reshape(hsv.shape)


        rgb = cv2.imread(path.format(i))
        # rgb[:90, :, :] = 0
        rgb = extract_edges(rgb)

        plt.clf()
        plt.imshow(rgb)
        plt.draw()
        plt.pause(1/60)


        # plt.imshow(mask)
