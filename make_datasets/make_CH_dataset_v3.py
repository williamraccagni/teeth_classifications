import csv
import cv2
import imutils as imutils
import numpy as np

def age_labels_return():

    path = './stadiazione_rx_opt_CH'

    with open((path + '/anni_CH.csv'), newline='') as f:
        reader = csv.reader(f)
        data = list(reader)

    return data


if __name__ == '__main__':


    # Preparazione Dataset
    path = './stadiazione_rx_opt_CH'
    numero_soggetti = 237

    nd_path = './stadiazione_rx_opt_CH_v3'

    # Load labels
    age_labels_list = age_labels_return()

    print(path)
    print(numero_soggetti)
    print(nd_path)
    print(len(age_labels_list))
    print(age_labels_list)

    for index in range(1, (numero_soggetti + 1) ):
        # Open csv
        with open((path + '/soggetto_' + str(index) + '.csv'), newline='') as f:
            reader = csv.reader(f)
            data = list(reader)[0]

        # update Dati
        data[9] = age_labels_list[index - 1][1]
        data[10] = age_labels_list[index - 1][2]
        data[11] = age_labels_list[index - 1][3]
        data[12] = age_labels_list[index - 1][4]

        # immagine Destra

        img_dx = cv2.imread(path + '/soggetto_' + str(index) + '_dx.bmp')
        cv2.imwrite(nd_path + '/soggetto_' + str(index) + '_dx.bmp', img_dx)

        # Immagine Sinistra

        img_sx = cv2.imread(path + '/soggetto_' + str(index) + '_sx.bmp')
        cv2.imwrite(nd_path + '/soggetto_' + str(index) + '_sx.bmp', img_sx)

        # Salva csv
        with open((nd_path + '/soggetto_' + str(index) + '.csv'), "w") as f:
            writer = csv.writer(f)
            writer.writerow(data)