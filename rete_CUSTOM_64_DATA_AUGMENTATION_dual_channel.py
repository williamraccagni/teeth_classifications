#64 custom DATA AUGMENTATION

import cv2
import numpy as np
import sklearn.model_selection
import tensorflow as tf
from tensorflow import keras

import time

from keras.callbacks import ReduceLROnPlateau

from tensorflow.keras import layers, models, Model, optimizers

import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots

from matplotlib import pyplot as plt

# def dual_channel_data_augmentation(X, Y):
#
#     X_simmetry = []
#     Y_simmetry = []
#
#     # Aggiungi coppie simmetriche
#     # *2
#     for index in range(len(Y)):
#
#         images_pair = X[index]
#         label = Y[index]
#
#         # aggiungo una copia originale
#
#         X_simmetry.append(images_pair)
#         Y_simmetry.append(label)
#
#         # visualizza
#         # plt.imshow(image)
#         # plt.show()
#         # plt.clf()
#
#         # creo e aggiungo coppia simmetrica
#
#         sim_dx_img = np.flip(images_pair[1], axis=1)
#         sim_sx_img = np.flip(images_pair[0], axis=1)
#
#         # # TEST dimensione
#         # print("test sim_dx_img:")
#         # print(sim_dx_img.shape)
#
#
#
#         X_simmetry.append([sim_dx_img, sim_sx_img])
#         Y_simmetry.append(label)
#
#
#     X_rotations = []
#     Y_rotations = []
#
#     # rotazioni 5 gradi
#     # per ogni coppia di immagini in X_simmetry aggiungi a X_rotations l'originale e le sue ruotate
#     # *3
#     for index in range(len(Y_simmetry)):
#
#         images_pair = X_simmetry[index]
#         label = Y_simmetry[index]
#
#         # aggiungo una copia originale
#
#         X_rotations.append(images_pair)
#         Y_rotations.append(label)
#
#         # grab the dimensions of the image and calculate the center of the image
#         (h, w) = images_pair[0].shape#[:2] #prendo l'immagine destra come riferimento
#         (cX, cY) = (w // 2, h // 2)
#
#         for degree in [x for x in range(-5, 6, 5) if x != 0]:
#             M = cv2.getRotationMatrix2D((cX, cY), degree, 1.0)
#             rotated_dx = cv2.warpAffine(images_pair[0], M, (w, h))
#             M = cv2.getRotationMatrix2D((cX, cY), (-1 * degree), 1.0)
#             rotated_sx = cv2.warpAffine(images_pair[1], M, (w, h))
#
#             # #TEST dimensione rotazione
#             # print("test dimensione rotazione:")
#             # print(rotated_dx.shape)
#
#             X_rotations.append([rotated_dx, rotated_sx])
#             Y_rotations.append(label)
#
#
#
#     X_result = []
#     Y_result = []
#
#     # RIMPICCIOLIMENTO E INGRANDIMENTO
#     # per ogni coppia di immagini in X_rotations aggiungi a X_result l'originale e le sue scalate
#     # * 7
#     for index in range(len(Y_rotations)):
#
#         images_pair = X_rotations[index]
#         label = Y_rotations[index]
#
#         X_result.append(images_pair)
#         Y_result.append(label)
#
#         (h, w) = images_pair[0].shape#[:2] # prendo l'immagine destra come riferimento
#
#         # RIMPICCIOLIMENTO
#         for scale_percent in range(70, 91, 10):
#
#             resized_width = int(images_pair[0].shape[1] * scale_percent / 100) # prendo l'immagine destra come riferimento
#             resized_height = int(images_pair[0].shape[0] * scale_percent / 100) # prendo l'immagine destra come riferimento
#             resized_dim = (resized_width, resized_height)
#
#             # resize images
#             resized_dx = cv2.resize(images_pair[0], resized_dim, interpolation=cv2.INTER_AREA)
#             resized_sx = cv2.resize(images_pair[1], resized_dim, interpolation=cv2.INTER_AREA)
#
#             # # TEST dimensione resized
#             # print("test dimensione resized_dx:")
#             # print(resized_dx.shape)
#
#             left = int((w - resized_width) / 2)
#             top = int((h - resized_height) / 2)
#             right = w - left - resized_width
#             bottom = h - top - resized_height
#
#             result_dx = cv2.copyMakeBorder(resized_dx, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)#[0, 0, 0])
#             result_sx = cv2.copyMakeBorder(resized_sx, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)#[0, 0, 0])
#
#             # # TEST dimensione result
#             # print("test dimensione result_dx:")
#             # print(result_dx.shape)
#
#             X_result.append([result_dx, result_sx])
#             Y_result.append(label)
#
#         # INGRANDIMENTO
#         for scale_percent in range(110, 131, 10):
#             resized_width = int(images_pair[0].shape[1] * scale_percent / 100) # prendo l'immagine destra come riferimento
#             resized_height = int(images_pair[0].shape[0] * scale_percent / 100) # prendo l'immagine destra come riferimento
#             resized_dim = (resized_width, resized_height)
#
#             # resize image
#             resized_dx = cv2.resize(images_pair[0], resized_dim, interpolation=cv2.INTER_AREA)
#             resized_sx = cv2.resize(images_pair[1], resized_dim, interpolation=cv2.INTER_AREA)
#
#             # # TEST dimensione resized
#             # print("test dimensione resized_dx:")
#             # print(resized_dx.shape)
#
#             left = int((resized_width - w) / 2)
#             top = int((resized_height - h) / 2)
#             right = resized_width - left - w
#             bottom = resized_height - top - h
#
#             result_dx = resized_dx[top:-bottom, left:-right]#, :]
#             result_sx = resized_sx[top:-bottom, left:-right]#, :]
#
#             # # TEST dimensione result
#             # print("test dimensione result_dx:")
#             # print(result_dx.shape)
#
#             X_result.append([result_dx, result_sx])
#             Y_result.append(label)
#
#     #Stampa immagini e controllo dimensione
#
#     print(X.shape)
#     print(len(X_simmetry))
#     print(len(X_rotations))
#     print(len(X_result))
#
#     print(Y.shape)
#     print(len(Y_simmetry))
#     print(len(Y_rotations))
#     print(len(Y_result))
#
#     # for index in range(len(X_result)):
#     #
#     #     # visualizza dx
#     #     plt.imshow(X_result[index][0], cmap='gray')
#     #     plt.show()
#     #     plt.clf()
#     #     # visualizza sx
#     #     plt.imshow(X_result[index][1], cmap='gray')
#     #     plt.show()
#     #     plt.clf()
#
#
#
#     return np.array(X_result), np.array(Y_result)

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

    test_size = 0.2
    validation_size = 0.25

    batch_size = 100
    epochs = 20


    # For reproducibility
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # ----Holdout Method----
    X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, original_labels, test_size=test_size,
                                                                                random_state=seed)
    # get subtrain and validation sets
    X_subtrain, X_validation, Y_subtrain, Y_validation = sklearn.model_selection.train_test_split(X_train, Y_train,
                                                                                                  test_size=validation_size,
                                                                                                  random_state=seed)

    print(Y_test)
    print(Y_train)
    print(Y_validation)
    print(Y_subtrain)

    print(Y_test.shape)
    print(Y_train.shape)
    print(Y_validation.shape)
    print(Y_subtrain.shape)



    print("Len of test set:")
    print(X_test.shape[0])
    print("Len of train set:")
    print(X_train.shape[0])
    print("Len of validation set:")
    print(X_validation.shape[0])
    print("Len of subtrain set:")
    print(X_subtrain.shape[0])



    print(X_test.shape)
    print(X_train.shape)
    print(X_validation.shape)
    print(X_subtrain.shape)

    # SUB TRAINING SET AUGMENTATION

    X_subtrain, Y_subtrain = dual_channel_data_augmentation(X_subtrain, Y_subtrain)
    print(Y_subtrain)
    print(Y_subtrain.shape)
    print("Len of train set after Data Augmentation:")
    print(X_subtrain.shape[0])
    print(X_subtrain.shape)

    # print('prova contenuto')
    # print(X_subtrain[0][0][32][32])
    # print(X_subtrain[0][1][32][32])


    # SET STACKS

    X_test = np.array([np.stack((X_test[index][0],X_test[index][1]), axis=-1)
                  for index in range(X_test.shape[0])])

    # X_train = np.array([np.stack((X_train[index][0], X_train[index][1]), axis=-1) # unire trainig e validation
    #                        for index in range(X_train.shape[0])])

    X_validation = np.array([np.stack((X_validation[index][0], X_validation[index][1]), axis=-1)
                           for index in range(X_validation.shape[0])])

    X_subtrain = np.array([np.stack((X_subtrain[index][0], X_subtrain[index][1]), axis=-1)
                           for index in range(X_subtrain.shape[0])])

    #UPDATE train con subtrain aumentato
    X_train = np.concatenate((X_subtrain, X_validation), axis=0)
    Y_train = np.concatenate((Y_subtrain, Y_validation), axis=0)



    # print('prova contenuto 2')
    # print(X_subtrain[0][32][32])

    print(Y_test.shape)
    print(Y_train.shape)
    print(Y_validation.shape)
    print(Y_subtrain.shape)

    print("Len of test set:")
    print(X_test.shape[0])
    print("Len of train set:")
    print(X_train.shape[0])
    print("Len of validation set:")
    print(X_validation.shape[0])
    print("Len of subtrain set:")
    print(X_subtrain.shape[0])

    print(X_test.shape)
    print(X_train.shape)
    print(X_validation.shape)
    print(X_subtrain.shape)



    # clear keras session
    tf.keras.backend.clear_session()

    # tempo iniziale
    start_time = time.time()

    # it trains the model for a fixed number of epochs
    history = model.fit(x=X_subtrain, y=Y_subtrain, validation_data=(X_validation, Y_validation), batch_size=batch_size,
              epochs=epochs)

    # tempo esecuzione
    print("--- %s secondi di training ---" % (time.time() - start_time))

    # ho aggiunto history

    # ------------------------------

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



    # # METRICS
    #
    # # predictions and labels
    # train_predictions = [int(round(x[0])) for x in model.predict(X_train)]
    # predictions = [int(round(x[0])) for x in model.predict(X_test)]
    #
    # print('Model scores (Holdout Method):')
    #
    # print('Confusion Matrix (test set):')
    # print(sklearn.metrics.confusion_matrix(y_true=Y_test, y_pred=predictions, labels=np.unique(Y_test)))
    #
    # print("Accuracy (train set):",
    #       sklearn.metrics.accuracy_score(y_true=Y_train, y_pred=train_predictions))
    # print("Accuracy (test set):",
    #       sklearn.metrics.accuracy_score(y_true=Y_test, y_pred=predictions))
    #
    # print('Classification results (test set):')
    # print(sklearn.metrics.classification_report(y_true=Y_test, y_pred=predictions))
    #
    # # Confusion Matrix Plot
    # matrix = sklearn.metrics.confusion_matrix(y_true=Y_test, y_pred=predictions, normalize='true')
    # print(matrix)
    #
    # sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=matrix).plot() #, display_labels=labels_map).plot()
    #
    # plt.title('Model Classifier Confusion Matrix\nHoldout Method, Test Set')
    # plt.show()
    # plt.clf()  # clear plot for next method

    # METRICS

    # predictions and labels
    train_predictions = [int(round(x[0])) for x in model.predict(X_train)]
    test_predictions = [int(round(x[0])) for x in model.predict(X_test)]

    print('Lunghezza finale etichette train:')
    print(len(train_predictions))
    print(len(Y_train))
    print('Lunghezza finale etichette test:')
    print(len(test_predictions))
    print(len(Y_test))

    print('Model scores (Holdout Method):')

    print('Confusion Matrix (train set):')
    print(sklearn.metrics.confusion_matrix(y_true=Y_train, y_pred=train_predictions, labels=np.unique(Y_train)))
    print('Confusion Matrix (test set):')
    print(sklearn.metrics.confusion_matrix(y_true=Y_test, y_pred=test_predictions, labels=np.unique(Y_test)))

    print("Accuracy (train set):",
          sklearn.metrics.accuracy_score(y_true=Y_train, y_pred=train_predictions))
    print("Accuracy (test set):",
          sklearn.metrics.accuracy_score(y_true=Y_test, y_pred=test_predictions))

    print('Classification results (train set):')
    print(sklearn.metrics.classification_report(y_true=Y_train, y_pred=train_predictions))
    print('Classification results (test set):')
    print(sklearn.metrics.classification_report(y_true=Y_test, y_pred=test_predictions))

    # Confusion Matrix Plot Train
    matrix = sklearn.metrics.confusion_matrix(y_true=Y_train, y_pred=train_predictions, normalize='all')
    print('Train Confusion Matrix')
    print(matrix)

    sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=matrix).plot() #, display_labels=labels_map).plot()

    plt.title('Model Classifier Confusion Matrix\nHoldout Method, Train Set')
    plt.show()
    plt.clf()  # clear plot for next method

    # Confusion Matrix Plot Test
    matrix = sklearn.metrics.confusion_matrix(y_true=Y_test, y_pred=test_predictions, normalize='all')
    print('Test Confusion Matrix')
    print(matrix)

    sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=matrix).plot()  # , display_labels=labels_map).plot()

    plt.title('Model Classifier Confusion Matrix\nHoldout Method, Test Set')
    plt.show()
    plt.clf()  # clear plot for next method
