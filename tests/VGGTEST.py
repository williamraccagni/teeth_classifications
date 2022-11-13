import csv
import cv2
import numpy as np
import sklearn
import sklearn.model_selection
import tensorflow as tf
from tensorflow import keras

def ch_labels():

    labels = []

    path = './stadiazione_rx_opt_CH'

    with open((path + '/anni_CH.csv'), newline='') as f:
        reader = csv.reader(f)
        data = list(reader)

    print(data)

    sog_list = [ [int(data[index][1]), int(data[index][2])] for index in range(len(data))]

    print(sog_list)

    for x in sog_list:
        if (x[0] == 14 and x[1] >= 7) or (x[0] == 15 and x[1] <= 6):
            labels.append(15)
            labels.append(15)
        if (x[0] == 15 and x[1] >= 7) or (x[0] == 16 and x[1] <= 6):
            labels.append(16)
            labels.append(16)
        if (x[0] == 16 and x[1] >= 7) or (x[0] == 17 and x[1] <= 6):
            labels.append(17)
            labels.append(17)
        if (x[0] == 17 and x[1] >= 7) or (x[0] == 18 and x[1] <= 6):
            labels.append(18)
            labels.append(18)
        if (x[0] == 18 and x[1] >= 7) or (x[0] == 19 and x[1] <= 6):
            labels.append(19)
            labels.append(19)
        if (x[0] == 19 and x[1] >= 7) or (x[0] == 20 and x[1] <= 6):
            labels.append(20)
            labels.append(20)
        if (x[0] == 20 and x[1] >= 7) or (x[0] == 21 and x[1] <= 6):
            labels.append(21)
            labels.append(21)
        if (x[0] == 21 and x[1] >= 7) or (x[0] == 22 and x[1] <= 6):
            labels.append(22)
            labels.append(22)

    print(len(labels))

    return labels

if __name__ == '__main__':


    # Preparazione Dataset
    path = './stadiazione_rx_opt_CH'
    numero_soggetti = 237

    nd_path = './test_CH_dataset_vgg'

    print(path)
    print(numero_soggetti)

    cornice = 40
    counter = 0

    for index in range(1, (numero_soggetti + 1) ):


        # immagine Destra

        img = cv2.imread(path + '/soggetto_'+str(index)+'_dx.bmp')

        img = img[cornice:-cornice,cornice:-cornice, :]

        img = cv2.resize(img, (224,224), interpolation=cv2.INTER_AREA)

        cv2.imwrite(nd_path + '/'+str(counter)+'.bmp', img)

        counter = counter + 1

        # print(img.shape)



        # immagine Sinistra

        img = cv2.imread(path + '/soggetto_' + str(index) + '_sx.bmp')

        img = img[cornice:-cornice,cornice:-cornice, :]

        img = np.flip(img,axis=1)

        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)

        cv2.imwrite(nd_path + '/'+str(counter)+'.bmp', img)

        counter = counter + 1

        # print(img.shape)


    # ETICHETTE

    labels = ch_labels()

    print(labels)
    with open(nd_path + '/labels.csv', 'w', newline='') as f:

        write = csv.writer(f)

        write.writerow(labels)




    # -----------------------------------------------------------------------------------------------------------------

    # Load Dataset

    numero_soggetti = 474

    X = np.array([cv2.imread(nd_path + '/' +str(index)+'.bmp')/255 for index in range(numero_soggetti)])
    original_labels = np.genfromtxt((nd_path + '/labels.csv'), delimiter=',').astype(int)

    Y = []
    for label in original_labels:
        if (label==15):
            Y.append(np.array([1,0,0,0,0,0,0,0]).astype(np.float32))
        if (label==16):
            Y.append(np.array([0,1,0,0,0,0,0,0]).astype(np.float32))
        if (label==17):
            Y.append(np.array([0,0,1,0,0,0,0,0]).astype(np.float32))
        if (label==18):
            Y.append(np.array([0,0,0,1,0,0,0,0]).astype(np.float32))
        if (label==19):
            Y.append(np.array([0,0,0,0,1,0,0,0]).astype(np.float32))
        if (label==20):
            Y.append(np.array([0,0,0,0,0,1,0,0]).astype(np.float32))
        if (label==21):
            Y.append(np.array([0,0,0,0,0,0,1,0]).astype(np.float32))
        if (label==22):
            Y.append(np.array([0,0,0,0,0,0,0,1]).astype(np.float32))

    Y = np.array(Y)

    print(X.shape)
    print(Y.shape)






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

    # Train and Metrics

    # For reproducibility
    np.random.seed(0)
    tf.random.set_seed(0)

    # ----Holdout Method----
    X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.2,
                                                                                random_state=0)
    # get subtrain and validation sets
    X_subtrain, X_validation, Y_subtrain, Y_validation = sklearn.model_selection.train_test_split(X_train, Y_train,
                                                                                                  test_size=0.2,
                                                                                                  random_state=0)

    # clear keras session
    tf.keras.backend.clear_session()

    # it trains the model for a fixed number of epochs
    model.fit(x=X_subtrain, y=Y_subtrain, validation_data=(X_validation, Y_validation), batch_size=100,
            epochs=500)





