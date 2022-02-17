import cv2
import numpy as np
import sklearn.model_selection
import tensorflow as tf
from tensorflow import keras

from matplotlib import pyplot as plt

import time

if __name__ == '__main__':

    path = './dataset_vgg_UP_DOWN_magmin_CH'
    numero_soggetti = 408

    # Load Dataset

    # vgg16 normalized images, X
    X = np.array( [ tf.keras.applications.vgg16.preprocess_input(cv2.imread(path + '/' + str(index) + '.bmp'))
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
