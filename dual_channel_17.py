import cv2
import numpy as np
import sklearn.model_selection
import tensorflow as tf
from tensorflow import keras

from keras.callbacks import ReduceLROnPlateau

from tensorflow.keras import layers, models, Model, optimizers

import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots

import time

from matplotlib import pyplot as plt

def dual_channel_data_augmentation(X, Y):

    X_simmetry = []
    Y_simmetry = []

    # Aggiungi coppie simmetriche
    # *4
    for index in range(len(Y)):

        images_pair = X[index]
        label = Y[index]

        # aggiungo una copia originale

        X_simmetry.append(images_pair)
        Y_simmetry.append(label)

        # visualizza
        # plt.imshow(image)
        # plt.show()
        # plt.clf()

        # creo e aggiungo coppia simmetrica

        sim_dx_img = np.flip(images_pair[1], axis=1)
        sim_sx_img = np.flip(images_pair[0], axis=1)

        # # TEST dimensione
        # print("test sim_dx_img:")
        # print(sim_dx_img.shape)



        X_simmetry.append([sim_dx_img, sim_sx_img])
        Y_simmetry.append(label)

        # creo e aggiungo coppia con denti specchaiti su loro stessi

        sim_dx_img = np.flip(images_pair[0], axis=1)
        sim_sx_img = np.flip(images_pair[1], axis=1)

        X_simmetry.append([sim_dx_img, sim_sx_img])
        Y_simmetry.append(label)


        # creo e aggiungo coppia con denti solo scambiati di posto

        X_simmetry.append([images_pair[1], images_pair[0]])
        Y_simmetry.append(label)


    X_rotations = []
    Y_rotations = []

    # rotazioni -6,-4,-2,2,4,6 gradi
    # per ogni coppia di immagini in X_simmetry aggiungi a X_rotations l'originale e le sue ruotate
    # *7
    for index in range(len(Y_simmetry)):

        images_pair = X_simmetry[index]
        label = Y_simmetry[index]

        # aggiungo una copia originale

        X_rotations.append(images_pair)
        Y_rotations.append(label)

        # grab the dimensions of the image and calculate the center of the image
        (h, w) = images_pair[0].shape#[:2] #prendo l'immagine destra come riferimento
        (cX, cY) = (w // 2, h // 2)

        for degree in [x for x in range(-6, 7, 2) if x != 0]:
            M = cv2.getRotationMatrix2D((cX, cY), degree, 1.0)
            rotated_dx = cv2.warpAffine(images_pair[0], M, (w, h))
            M = cv2.getRotationMatrix2D((cX, cY), (-1 * degree), 1.0)
            rotated_sx = cv2.warpAffine(images_pair[1], M, (w, h))

            # #TEST dimensione rotazione
            # print("test dimensione rotazione:")
            # print(rotated_dx.shape)

            X_rotations.append([rotated_dx, rotated_sx])
            Y_rotations.append(label)



    #Stampa immagini e controllo dimensione

    print(X.shape)
    print(len(X_simmetry))
    print(len(X_rotations))

    print(Y.shape)
    print(len(Y_simmetry))
    print(len(Y_rotations))

    # for index in range(len(X_result)):
    #
    #     # visualizza dx
    #     plt.imshow(X_result[index][0], cmap='gray')
    #     plt.show()
    #     plt.clf()
    #     # visualizza sx
    #     plt.imshow(X_result[index][1], cmap='gray')
    #     plt.show()
    #     plt.clf()



    return np.array(X_rotations), np.array(Y_rotations)

if __name__ == '__main__':

    path = './dual_channel_64_CH'
    numero_soggetti = 218

    # Load Dataset

    # X
    X = np.array( [[cv2.imread(path + '/' + str(index) + '_dx.bmp', 0), cv2.imread(path + '/' + str(index) + '_sx.bmp', 0)]
         for index in range(numero_soggetti)] )


    # labels
    original_labels = np.genfromtxt((path + '/labels.csv'), delimiter=',').astype(int)



    print("Number of images:")
    print(X.shape)
    print("original_labels len:")
    print(original_labels.shape)
    print(original_labels.dtype)

    # ------------------------------------------------------------

    # MODELLO

    model = keras.Sequential([
        keras.layers.Conv2D(filters=32, kernel_size=(5, 5), strides=1, activation='relu', input_shape=(64, 64, 2)),
        # keras.layers.Conv2D(filters=32, kernel_size=(5, 5), strides=1, activation='relu'),
        keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
        keras.layers.Conv2D(filters=64, kernel_size=(5, 5), strides=1, activation='relu'),
        # keras.layers.Conv2D(filters=64, kernel_size=(5, 5), strides=1, activation='relu'),
        keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
        keras.layers.Flatten(),
        # output layer
        # keras.layers.Dense(10, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.summary()

    # -------------------------------------------------------------------------------------

    # TRAINING

    seed = 0



    batch_size = 100
    # modificato
    epochs = 100



    # For reproducibility
    np.random.seed(seed)
    tf.random.set_seed(seed)

    Y = original_labels

    print(Y)
    print(Y.shape)
    print("Len of X:")
    print(X.shape[0])
    print(X.shape)

    # TRAINING SET AUGMENTATION

    print('DATA AUGMENTATION:')

    print('during process')

    X, Y = dual_channel_data_augmentation(X, Y)

    print('after process')

    print(Y)
    print(Y.shape)
    print("Len of train set after Data Augmentation:")
    print(X.shape[0])
    print(X.shape)

    # SET STACKS

    print('SET STACK')

    X = np.array([np.stack((X[index][0], X[index][1]), axis=-1)
                       for index in range(X.shape[0])])

    print(Y.shape)

    print("Len of X:")
    print(X.shape[0])

    print(X.shape)

    # clear keras session
    tf.keras.backend.clear_session()

    # tempo iniziale
    start_time = time.time()

    # it trains the model for a fixed number of epochs, metto validation split
    history = model.fit(x=X, y=Y, validation_split=0.1,
                        batch_size=batch_size,
                        epochs=epochs)

    # tempo esecuzione
    print("--- %s secondi di training ---" % (time.time() - start_time))

    # ho aggiunto history

    # ------------------------------

    # GDD

    # GDD

    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    plt.clf()  # clear plot for next method
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    plt.clf()  # clear plot for next method




    # CARICO TEST MI

    test_path = './dual_channel_MI_17'
    X_test = [ [index, np.stack((cv2.imread(test_path + '/' + str(index) + '_dx.bmp', 0),
                                 cv2.imread(test_path + '/' + str(index) + '_sx.bmp', 0)), axis=-1)]
               for index in range(127, (173 + 1) )]

    for image_test in X_test:
        prediction = int(round(model.predict(np.array([image_test[1]]))[0][0]))
        print(image_test[0], prediction)
