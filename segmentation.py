import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from tqdm import tqdm


def segmentation(image):
    ret, thresh = cv2.threshold(
        image, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # add outline
    laplacian = cv2.Laplacian(image, cv2.CV_8U, ksize=5)
    ret, line = cv2.threshold(
        laplacian, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    line = cv2.medianBlur(line, ksize=3)
    thresh = cv2.bitwise_or(thresh, thresh, mask=line)

    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # make distance image
    tmp = cv2.convertScaleAbs(opening, alpha=(255))
    dist_transform = cv2.distanceTransform(tmp, cv2.DIST_L2, 5)

    # Finding sure foreground area
    ret, sure_fg = cv2.threshold(
        dist_transform, 0.05 * dist_transform.max(), 1, 0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    sure_bg = np.uint8(sure_bg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    tmp = cv2.convertScaleAbs(image, alpha=(255 / 65535))
    color = cv2.cvtColor(tmp, cv2.COLOR_GRAY2BGR)
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
    # import copy

    filename_list = list(pathlib.Path(
        'data/20201219_073658/depth').glob('*.png'))

    # fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    # ax[0].set_title('image')
    # ax[1].set_title('segmentation')

    # image_list = []
    # markers_list = []
    for filename in tqdm(filename_list):
        image = cv2.imread(
            'data/20201219_073658/depth/{}'.format(filename.name),
            cv2.IMREAD_ANYDEPTH)
        # image = cv2.convertScaleAbs(image, alpha=(255 / 65535))
        markers = segmentation(image)
        # markers_origin = copy.deepcopy(markers)

        markers_color = cv2.convertScaleAbs(
            markers, alpha=(255 / np.max(markers)))
        markers_color = cv2.applyColorMap(markers_color, cv2.COLORMAP_RAINBOW)
        markers_color[markers == 1] = [0, 0, 0]
        # markers_color[markers == 1] = [255, 255, 255]
        cv2.imwrite('tmp/{}'.format(filename.name), markers_color)

        # image_list.append(image)
        # markers_list.append(markers)

        # ax[0].imshow(image)
        # ax[1].imshow(markers, cmap='tab20')
        # # plt.pause(0.02)
        # plt.savefig('tmp/{}'.format(filename.name))

    # ani = AnimationMaker(image_list, markers_list).makeAnimation()
    # ani.save('animation.gif', writer='pillow')
