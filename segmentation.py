import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from tqdm import tqdm


def segmentation(image):
    ret, thresh = cv2.threshold(
        image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

    # Finding sure foreground area
    ret, sure_fg = cv2.threshold(
        dist_transform, 0.05 * dist_transform.max(), 255, 0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(color, markers)

    return markers


class AnimationMaker():
    def __init__(self, image_list, markers_list, fps=60):
        self.image_list = image_list
        self.markers_list = markers_list
        self.interval = 1000 / fps
        self._init_figure()

    def _init_figure(self):
        self.fig, self.ax = plt.subplots(1, 2, figsize=(12, 6))
        self.ax[0].set_title('image')
        self.ax[1].set_title('segmentation')

    def _update(self, i):
        self.ax[0].imshow(self.image_list[i], cmap='jet')
        self.ax[1].imshow(self.markers_list[i], cmap='tab20')
        print('{}/{}'.format(i, len(self.image_list)))

    def makeAnimation(self):
        return animation.FuncAnimation(self.fig, self._update,
                                       interval=self.interval, frames=len(self.image_list))


if __name__ == '__main__':
    import pathlib

    filename_list = list(pathlib.Path(
        'data/20201219_073658/depth').glob('*.png'))
    # for p in filename_list:
    #     print(p.name)

    # image = cv2.imread('data/20201219_073658/depth/depth023307409300.png', 0)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].set_title('image')
    ax[1].set_title('segmentation')

    # image_list = []
    # markers_list = []
    for filename in tqdm(filename_list):
        image = cv2.imread('data/20201219_073658/depth/' + filename.name, 0)
        markers = segmentation(image)

        # image_list.append(image)
        # markers_list.append(markers)

        ax[0].imshow(image, cmap='jet')
        ax[1].imshow(markers, cmap='tab20')
        plt.pause(0.02)

    # ani = AnimationMaker(image_list, markers_list).makeAnimation()
    # ani.save('animation.gif', writer='pillow')
