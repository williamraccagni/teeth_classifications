import cv2
import numpy as np
import sklearn.model_selection
import tensorflow as tf
from tensorflow import keras

import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots

from matplotlib import pyplot as plt

def one_hot_encoding_CH(label : int) -> np.array:
    if (label == 15):
        return np.array([1, 0, 0, 0, 0, 0, 0, 0]).astype(np.float32)
    if (label == 16):
        return np.array([0, 1, 0, 0, 0, 0, 0, 0]).astype(np.float32)
    if (label == 17):
        return np.array([0, 0, 1, 0, 0, 0, 0, 0]).astype(np.float32)
    if (label == 18):
        return np.array([0, 0, 0, 1, 0, 0, 0, 0]).astype(np.float32)
    if (label == 19):
        return np.array([0, 0, 0, 0, 1, 0, 0, 0]).astype(np.float32)
    if (label == 20):
        return np.array([0, 0, 0, 0, 0, 1, 0, 0]).astype(np.float32)
    if (label == 21):
        return np.array([0, 0, 0, 0, 0, 0, 1, 0]).astype(np.float32)
    if (label == 22):
        return np.array([0, 0, 0, 0, 0, 0, 0, 1]).astype(np.float32)


if __name__ == '__main__':

    path = './dataset_vgg_CH'
    numero_soggetti = 413

    # Load Dataset

    # vgg16 normalized images, X
    X = np.array( [ tf.keras.applications.vgg16.preprocess_input(cv2.imread(path + '/' + str(index) + '.bmp'))
                    for index in range(numero_soggetti)] )

    # labels
    original_labels = np.genfromtxt((path + '/labels.csv'), delimiter=',').astype(int)

    # Y = []
    # for label in original_labels:
    #     if (label == 15):
    #         Y.append(np.array([1, 0, 0, 0, 0, 0, 0, 0]).astype(np.float32))
    #     if (label == 16):
    #         Y.append(np.array([0, 1, 0, 0, 0, 0, 0, 0]).astype(np.float32))
    #     if (label == 17):
    #         Y.append(np.array([0, 0, 1, 0, 0, 0, 0, 0]).astype(np.float32))
    #     if (label == 18):
    #         Y.append(np.array([0, 0, 0, 1, 0, 0, 0, 0]).astype(np.float32))
    #     if (label == 19):
    #         Y.append(np.array([0, 0, 0, 0, 1, 0, 0, 0]).astype(np.float32))
    #     if (label == 20):
    #         Y.append(np.array([0, 0, 0, 0, 0, 1, 0, 0]).astype(np.float32))
    #     if (label == 21):
    #         Y.append(np.array([0, 0, 0, 0, 0, 0, 1, 0]).astype(np.float32))
    #     if (label == 22):
    #         Y.append(np.array([0, 0, 0, 0, 0, 0, 0, 1]).astype(np.float32))
    #
    # Y = np.array(Y)
    #
    # print(X.shape)
    # print(Y.shape)

    print("Number of images:")
    print(X.shape)
    print("original_labels len:")
    print(len(original_labels))

    # ------------------------------------------------------------

    # MODELLO

    vgg_model = tf.keras.applications.VGG16(input_shape=[224, 224] + [3], weights='imagenet', include_top=False)

    # freeze layers
    vgg_model.trainable = False

    # Create Model
    model = keras.Sequential([
        # pre trained layer
        vgg_model,
        # Flatten
        keras.layers.Flatten(),
        # output layer
        keras.layers.Dense(8, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.summary()

    # -------------------------------------------------------------------------------------

    # TRAINING

    seed = 0

    test_size = 0.2
    validation_size = 0.25

    batch_size = 100
    epochs = 100


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
    print(Y_validation)
    print(Y_subtrain)

    # Conversion of Ys to one-hot encoding

    Y_test = np.array([one_hot_encoding_CH(y) for y in Y_test])
    Y_validation = np.array([one_hot_encoding_CH(y) for y in Y_validation])
    Y_subtrain = np.array([one_hot_encoding_CH(y) for y in Y_subtrain])

    print(Y_test)
    print(Y_validation)
    print(Y_subtrain)



    print("Len of test set:")
    print(X_test.shape[0])
    print("Len of validation set:")
    print(X_validation.shape[0])
    print("Len of train set:")
    print(X_subtrain.shape[0])


    # clear keras session
    tf.keras.backend.clear_session()

    # it trains the model for a fixed number of epochs
    history = model.fit(x=X_subtrain, y=Y_subtrain, validation_data=(X_validation, Y_validation), batch_size=batch_size,
              epochs=epochs)

    # ho aggiunto history

    # ------------------------------

    # METRICS

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