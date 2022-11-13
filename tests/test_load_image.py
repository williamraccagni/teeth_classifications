import cv2
import numpy as np


from matplotlib import pyplot as plt

if __name__ == '__main__':

    path = './dataset_vgg_UP_DOWN_magmin_CH'

    img = cv2.imread(path + '/7.bmp')

    print(img.shape)
    for y in range(224):
        for x in range(224):
            print(img[y][x])

    # visualizza
    plt.imshow(img)
    plt.show()
    plt.clf()