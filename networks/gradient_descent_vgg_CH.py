import pathlib
import shutil
import tempfile

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


def get_optimizer():
  return tf.keras.optimizers.Adam(lr_schedule)

def get_callbacks(name):
  return [
    tfdocs.modeling.EpochDots(),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=200),
    tf.keras.callbacks.TensorBoard(logdir/name),
  ]

def compile_and_fit(model, name, optimizer=None, max_epochs=10000):
  if optimizer is None:
    optimizer = get_optimizer()
  model.compile(optimizer=optimizer,
                loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                metrics=[
                  tf.keras.losses.CategoricalCrossentropy(
                      from_logits=True, name='categorical_crossentropy'),
                  'accuracy'])

  model.summary()

  history = model.fit(
    x=X_subtrain, y=Y_subtrain,
    validation_data=(X_validation, Y_validation),
    steps_per_epoch = STEPS_PER_EPOCH,
    epochs=max_epochs,
    callbacks=get_callbacks(name),
    verbose=0)
  return history

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

    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #
    # model.summary()

    # -------------------------------------------------------------------------------------

    # TRAINING

    seed = 0

    test_size = 0.2
    validation_size = 0.2

    batch_size = 100
    epochs = 10


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


    # clear keras session
    tf.keras.backend.clear_session()

    # # it trains the model for a fixed number of epochs
    # model.fit(x=X_subtrain, y=Y_subtrain, validation_data=(X_validation, Y_validation), batch_size=batch_size,
    #           epochs=epochs)


    # ------------------------------
    #-------------------------------

    # GD

    logdir = pathlib.Path(tempfile.mkdtemp()) / "tensorboard_logs"
    shutil.rmtree(logdir, ignore_errors=True)

    N_VALIDATION = X_validation.shape[0]
    N_TRAIN = X_subtrain.shape[0]
    BUFFER_SIZE = X_subtrain.shape[0]
    BATCH_SIZE = 200
    STEPS_PER_EPOCH = N_TRAIN // BATCH_SIZE

    print(N_VALIDATION)
    print(N_TRAIN)

    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
        0.001,
        decay_steps=STEPS_PER_EPOCH * 1000,
        decay_rate=1,
        staircase=False)

    step = np.linspace(0, 100000)
    lr = lr_schedule(step)
    plt.figure(figsize=(8, 6))
    plt.plot(step / STEPS_PER_EPOCH, lr)
    plt.ylim([0, max(plt.ylim())])
    plt.xlabel('Epoch')
    _ = plt.ylabel('Learning Rate')
    plt.show()



    size_histories = {}

    size_histories['Model'] = compile_and_fit(model, 'sizes/Model')

    # smoothing_std????
    plotter = tfdocs.plots.HistoryPlotter(metric='categorical_crossentropy', smoothing_std=10)

    plotter.plot(size_histories)
    plt.ylim([0.0, 1.0])
    plt.show()