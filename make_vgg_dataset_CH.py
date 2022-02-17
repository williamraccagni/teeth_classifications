import csv
import cv2
import numpy as np

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
        if (x[0] == 15 and x[1] >= 7) or (x[0] == 16 and x[1] <= 6):
            labels.append(16)
        if (x[0] == 16 and x[1] >= 7) or (x[0] == 17 and x[1] <= 6):
            labels.append(17)
        if (x[0] == 17 and x[1] >= 7) or (x[0] == 18 and x[1] <= 6):
            labels.append(18)
        if (x[0] == 18 and x[1] >= 7) or (x[0] == 19 and x[1] <= 6):
            labels.append(19)
        if (x[0] == 19 and x[1] >= 7) or (x[0] == 20 and x[1] <= 6):
            labels.append(20)
        if (x[0] == 20 and x[1] >= 7) or (x[0] == 21 and x[1] <= 6):
            labels.append(21)
        if (x[0] == 21 and x[1] >= 7) or (x[0] == 22 and x[1] <= 6):
            labels.append(22)

    print(len(labels))

    return labels

if __name__ == '__main__':


    # Preparazione Dataset
    path = './stadiazione_rx_opt_CH'
    numero_soggetti = 237

    nd_path = './dataset_vgg_CH'

    # Load labels
    original_labels = ch_labels()
    labels = []

    print(path)
    print(numero_soggetti)
    print(nd_path)
    print(len(original_labels))
    print(original_labels)

    cornice = 40
    counter = 0



    for index in range(1, (numero_soggetti + 1) ):

        # Open csv
        with open((path + '/soggetto_' + str(index) + '.csv'), newline='') as f:
            reader = csv.reader(f)
            data = list(reader)[0]

        # immagine Destra

        if data[1] == 'dente presente':

            img = cv2.imread(path + '/soggetto_'+str(index)+'_dx.bmp')

            img = img[cornice:-cornice,cornice:-cornice, :]

            img = cv2.resize(img, (224,224), interpolation=cv2.INTER_AREA)

            cv2.imwrite(nd_path + '/'+str(counter)+'.bmp', img)

            counter = counter + 1

            labels.append(original_labels[(index - 1)])

            # print(img.shape)



        # immagine Sinistra

        if data[4] == 'dente presente':

            img = cv2.imread(path + '/soggetto_' + str(index) + '_sx.bmp')

            img = img[cornice:-cornice,cornice:-cornice, :]

            img = np.flip(img,axis=1)

            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)

            cv2.imwrite(nd_path + '/'+str(counter)+'.bmp', img)

            counter = counter + 1

            labels.append(original_labels[(index - 1)])

            # print(img.shape)






    # ETICHETTE

    print(labels)
    print(len(labels))

    with open(nd_path + '/labels.csv', 'w', newline='') as f:

        write = csv.writer(f)
        write.writerow(labels)