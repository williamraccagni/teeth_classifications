# single channel

import csv
import cv2
import numpy as np

def label_converter_magmin(years, months):

    if years >= 18:
        return 1
    else:
        return 0

    return 'error'

if __name__ == '__main__':

    IMG_DIM = 28

    # Preparazione Dataset
    path = './UP_DOWN_stadiazione_CH_gimp'
    numero_soggetti = 237

    nd_path = 'siamese_CH_dataset'

    original_labels = []
    labels = []

    print(path)
    print(numero_soggetti)
    print(nd_path)

    counter = 0

    for index in range(1, (numero_soggetti + 1) ):

        # Open csv
        with open((path + '/soggetto_' + str(index) + '.csv'), newline='') as f:
            reader = csv.reader(f)
            data = list(reader)[0]

        cur_label = label_converter_magmin( int(data[9]), int(data[10]))
        if(cur_label != 'error'):
            original_labels.append(cur_label)
        else:
            print('ERRORE')



        # immagine Destra

        if data[1] == 'dente presente' and (data[2] != 'sconosciuto' or data[3] != 'sconosciuto'):

            img = cv2.imread(path + '/soggetto_'+str(index)+'_dx.bmp',0)

            img = cv2.resize(img, (IMG_DIM, IMG_DIM), interpolation=cv2.INTER_AREA)

            cv2.imwrite(nd_path + '/'+str(counter)+'.bmp', img)

            counter = counter + 1

            labels.append(cur_label)

            # print(img.shape)




        # immagine Sinistra

        if data[4] == 'dente presente' and (data[5] != 'sconosciuto' or data[6] != 'sconosciuto'):

            img = cv2.imread(path + '/soggetto_' + str(index) + '_sx.bmp',0)

            img = cv2.resize(img, (IMG_DIM, IMG_DIM), interpolation=cv2.INTER_AREA)

            cv2.imwrite(nd_path + '/'+str(counter)+'.bmp', img)

            counter = counter + 1

            labels.append(cur_label)

            # print(img.shape)




    # ETICHETTE

    print(original_labels)
    print(len(original_labels))
    print(labels)
    print(len(labels))

    with open(nd_path + '/labels.csv', 'w', newline='') as f:

        write = csv.writer(f)
        write.writerow(labels)