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

from matplotlib import pyplot as plt

if __name__ == '__main__':

    path = './dataset_vgg_UP_DOWN_magmin_CH'
    numero_soggetti = 408

    # Load Dataset

    # vgg16 normalized images, X
    X = np.array([tf.keras.applications.vgg16.preprocess_input(cv2.imread(path + '/' + str(index) + '.bmp'))
                  for index in range(numero_soggetti)])

    # labels
    original_labels = np.genfromtxt((path + '/labels.csv'), delimiter=',').astype(int)



    print("Number of images:")
    print(X.shape)
    print("original_labels len:")
    print(original_labels.shape)
    print(original_labels.dtype)

    # ------------------------------------------------------------

    # MODELLO

    vgg_model = tf.keras.applications.VGG16(input_shape=(224, 224, 3), weights='imagenet', include_top=False)

    # Freeze four convolution blocks
    for layer in vgg_model.layers[:15]:
        layer.trainable = False
    # Make sure you have frozen the correct layers
    for i, layer in enumerate(vgg_model.layers):
        print(i, layer.name, layer.trainable)

    x = vgg_model.output
    x = keras.layers.Flatten()(x)  # Flatten dimensions to for use in FC layers
    x = keras.layers.Dense(1, activation='sigmoid')(x)

    model = keras.Model(inputs=vgg_model.input, outputs=x)

    lr_reduce = ReduceLROnPlateau(monitor='val_accuracy', factor=0.6, patience=8, verbose=1, mode='max', min_lr=5e-5)
    checkpoint = keras.callbacks.ModelCheckpoint('vgg16_finetune.h15', monitor='val_accuracy', mode='max', save_best_only=True,
                                 verbose=1)

    learning_rate = 5e-5

    model.compile(loss="binary_crossentropy", optimizer=optimizers.Adam(lr=learning_rate),
                           metrics=["accuracy"])

    model.summary()

    # Make sure you have frozen the correct layers
    for i, layer in enumerate(model.layers):
        print(i, layer.name, layer.trainable)

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
    print(Y_validation)
    print(Y_subtrain)

    print("Len of test set:")
    print(X_test.shape[0])
    print("Len of validation set:")
    print(X_validation.shape[0])
    print("Len of train set:")
    print(X_subtrain.shape[0])

    print(type(X_subtrain))
    print(type(X_validation))
    print(type(X_test))

    print(X_subtrain.dtype)
    print(X_validation.dtype)
    print(X_test.dtype)





    # # DATA NORMALIZATION
    #
    # X_subtrain = np.array([tf.keras.applications.vgg16.preprocess_input(x) for x in X_subtrain])
    # X_validation = np.array([tf.keras.applications.vgg16.preprocess_input(x) for x in X_validation])
    # X_test = np.array([tf.keras.applications.vgg16.preprocess_input(x) for x in X_test])
    #
    # print("Len of test set:")
    # print(X_test.shape[0])
    # print("Len of validation set:")
    # print(X_validation.shape[0])
    # print("Len of train set:")
    # print(X_subtrain.shape[0])
    #
    # print(type(X_subtrain))
    # print(type(X_validation))
    # print(type(X_test))
    #
    # print(X_subtrain.dtype)
    # print(X_validation.dtype)
    # print(X_test.dtype)




    # DATA AUGMENTATION

    # Augment images
    train_datagen = keras.preprocessing.image.ImageDataGenerator(zoom_range=0.2, rotation_range=5, horizontal_flip = True)
    # Fit augmentation to training images
    train_generator = train_datagen.flow(X_subtrain, Y_subtrain, batch_size=1)







    # clear keras session
    tf.keras.backend.clear_session()

    history = model.fit_generator(train_generator, validation_data=(X_validation,Y_validation), epochs=50, shuffle=True, callbacks=[lr_reduce],verbose=1)

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
    predictions = [int(round(x[0])) for x in model.predict(X_test)]

    print('Model scores (Holdout Method):')

    print('Confusion Matrix (test set):')
    print(sklearn.metrics.confusion_matrix(y_true=Y_test, y_pred=predictions, labels=np.unique(Y_test)))

    print("Accuracy (train set):",
          sklearn.metrics.accuracy_score(y_true=Y_train, y_pred=train_predictions))
    print("Accuracy (test set):",
          sklearn.metrics.accuracy_score(y_true=Y_test, y_pred=predictions))

    print('Classification results (test set):')
    print(sklearn.metrics.classification_report(y_true=Y_test, y_pred=predictions))

    # Confusion Matrix Plot
    matrix = sklearn.metrics.confusion_matrix(y_true=Y_test, y_pred=predictions, normalize='true')
    print(matrix)

    sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=matrix).plot()  # , display_labels=labels_map).plot()

    plt.title('Model Classifier Confusion Matrix\nHoldout Method, Test Set')
    plt.show()
    plt.clf()  # clear plot for next method