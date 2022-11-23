import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

GREEN = (60, 100)
BLACK = (0, 50)


class Controller:
    def get_control(self, im):
        im[260:, :] = 0
        blacks = self.colormask(im, BLACK)
        greens = self.colormask(im, GREEN)
        # return blacks, greens
        cblack = np.sum(blacks.flatten())
        cgreen = np.sum(greens.flatten())

        if cblack > cgreen:
            return 0.3
        else:
            return 0.7

    def colormask(self, im, band):
        colormin, colormask = band
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


        axs[0].clear()
        axs[0].imshow(rgb)


        steering = controller.get_control(rgb)
        # print(blacks, greens)
        print(steering)
        # if not isinstance(vector, int):
        #     axs[1].clear()
        #     axs[1].quiver(0, 0, *vector, scale=3)


        plt.draw()
        plt.pause(1 / 30)
        i += 30

