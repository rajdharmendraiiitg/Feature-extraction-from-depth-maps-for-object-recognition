import matplotlib.pyplot as plt

from skimage.feature import hog
from skimage import data, exposure
import cv2
import numpy as np


def hog_calculator(image):
    # image = data.astronaut()
    image = cv2.imread(image)
    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                        cells_per_block=(1, 1), visualize=True, multichannel=True)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(8, 4), sharex=True, sharey=True)

    ax1.axis('off')
    ax1.imshow(image, cmap=plt.cm.gray)
    ax1.set_title('Input image')

    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(
        hog_image, in_range=(0, 10))

    # hist_hog = cv2.calcHist([hog_image_rescaled], [0], None, [256], [0, 256])
    p = np.array(hog_image_rescaled).reshape((1, -1)).tolist()
    bins_range = (0, 180)
    bin_num = 228
    n, bins, patches = plt.hist(
        p, bin_num, facecolor='blue', alpha=0.5)

    print(len(n))

    # ax2.axis('off')
    # ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    # ax2.set_title('Histogram of Oriented Gradients')
    # plt.show()

    print(hog_image)
    print(n)
    return n


if __name__ == '__main__':
    hog_calculator('a.png')
