import cv2
import numpy as np
import sklearn.model_selection
import tensorflow as tf
from tensorflow import keras

import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots

from matplotlib import pyplot as plt

def images_augmentation (X, Y):

    X_partial = []
    Y_partial = []

    # ROTAZIONI
    # per ogni immagine in X aggiungi a X_partial l'originale e le sue ruotate
    for index in range(len(Y)):

        image = X[index]
        label = Y[index]

        X_partial.append(image)
        Y_partial.append(label)

        # visualizza
        # plt.imshow(image)
        # plt.show()
        # plt.clf()


        # grab the dimensions of the image and calculate the center of the image
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)

        for degree in [x for x in range(-40, 41, 5) if x != 0]:
            M = cv2.getRotationMatrix2D((cX, cY), degree, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h))

            X_partial.append(rotated)
            Y_partial.append(label)



    X_result = []
    Y_result = []

    # RIMPICCIOLIMENTO E INGRANDIMENTO
    # per ogni immagine in X_partial aggiungi a X_result l'originale e le sue scalate
    for index in range(len(Y_partial)):

        image = X_partial[index]
        label = Y_partial[index]

        X_result.append(image)
        Y_result.append(label)

        (h, w) = image.shape[:2]

        for scale_percent in range(70, 91, 10):

            resized_width = int(image.shape[1] * scale_percent / 100)
            resized_height = int(image.shape[0] * scale_percent / 100)
            resized_dim = (resized_width, resized_height)

            # resize image
            resized = cv2.resize(image, resized_dim, interpolation=cv2.INTER_AREA)

            # ht, wd, cc = resized.shape
            #
            # # create new image of desired size and color (blue) for padding
            # ww = image.shape[1]
            # hh = image.shape[0]
            # color = (0, 0, 0) #RGB
            # result = np.full((hh, ww, cc), color, dtype=np.uint8)
            #
            # print(image.dtype)
            # print(resized.dtype)
            # print(result.dtype)
            #
            # # compute center offset
            # xx = (ww - wd) // 2
            # yy = (hh - ht) // 2
            #
            # # copy img image into center of result image
            # result[yy:yy + ht, xx:xx + wd] = resized

            left = int( (w - resized_width) / 2)
            top = int( (h - resized_height) / 2)
            right = w - left - resized_width
            bottom = h - top - resized_height

            result = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

            X_result.append(result)
            Y_result.append(label)



        # INGRANDIMENTO

        for scale_percent in range(110, 131, 10):

            resized_width = int(image.shape[1] * scale_percent / 100)
            resized_height = int(image.shape[0] * scale_percent / 100)
            resized_dim = (resized_width, resized_height)

            # resize image
            resized = cv2.resize(image, resized_dim, interpolation=cv2.INTER_AREA)

            left = int((resized_width - w) / 2)
            top = int((resized_height - h) / 2)
            right = resized_width - left - w
            bottom = resized_height - top - h

            result = resized[top:-bottom,left:-right, :]

            X_result.append(result)
            Y_result.append(label)



    return np.array(X_result), np.array(Y_result)


if __name__ == '__main__':

    path = './dataset_vgg_CH'
    numero_soggetti = 413

    # Load Dataset

    # vgg16 normalized images, X
    X = np.array([cv2.imread(path + '/' + str(index) + '.bmp') for index in range(numero_soggetti)])

    # labels
    original_labels = np.genfromtxt((path + '/labels.csv'), delimiter=',').astype(int)


    print(type(X[0]))
    print(type(original_labels))
    print("Number of images:")
    print(X.shape)
    print("original_labels len:")
    print(len(original_labels))

    # ------------------------------------------------------------

    X_augmented, Y_augmented = images_augmentation(X, original_labels)

    print(X_augmented.shape)
    print(Y_augmented.shape)