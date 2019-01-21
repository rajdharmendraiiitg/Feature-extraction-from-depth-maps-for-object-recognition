import cv2
import numpy as np
from matplotlib import pyplot as plt


def get_pixel(img, center, x, y):
    new_value = 0
    try:
        if img[x][y] >= center:
            new_value = 1
    except:
        pass
    return new_value


def lbp_calculated_pixel(img, x, y):

    center = img[x][y]
    val_ar = []
    val_ar.append(get_pixel(img, center, x - 1, y + 1))     # top_right
    val_ar.append(get_pixel(img, center, x, y + 1))       # right
    val_ar.append(get_pixel(img, center, x + 1, y + 1))     # bottom_right
    val_ar.append(get_pixel(img, center, x + 1, y))       # bottom
    val_ar.append(get_pixel(img, center, x + 1, y - 1))     # bottom_left
    val_ar.append(get_pixel(img, center, x, y - 1))       # left
    val_ar.append(get_pixel(img, center, x - 1, y - 1))     # top_left
    val_ar.append(get_pixel(img, center, x - 1, y))       # top

    power_val = [1, 2, 4, 8, 16, 32, 64, 128]
    val = 0
    for i in range(len(val_ar)):
        val += val_ar[i] * power_val[i]
    return val


def lbp_calculator(image_filename):
    image_file = image_filename
    img_bgr = cv2.imread(image_file)
    height, width, channel = img_bgr.shape
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    img_lbp = np.zeros((height, width, 3), np.uint8)
    for i in range(0, height):
        for j in range(0, width):
            img_lbp[i, j] = lbp_calculated_pixel(img_gray, i, j)
    hist_lbp = cv2.calcHist([img_lbp], [0], None, [228], [0, 228])

    print(len(hist_lbp))

    return hist_lbp

if __name__ == '__main__':
    lbp_calculator('12.png')
