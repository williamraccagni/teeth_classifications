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

    nd_path = './UP_DOWN_stadiazione_CH'

    # Load labels
    age_labels_list = age_labels_return()

    print(path)
    print(numero_soggetti)
    print(nd_path)
    print(len(age_labels_list))
    print(age_labels_list)

    dim_finale = 290

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

        img = cv2.imread(path + '/soggetto_' + str(index) + '_dx.bmp')

        rotated = imutils.rotate_bound(img, -45)

        rot_dim = rotated.shape[0]
        bordo_a = int((rot_dim - dim_finale) / 2)
        bordo_b = rot_dim - dim_finale - bordo_a

        rotated = rotated[bordo_a:-bordo_b,bordo_a:-bordo_b, :]

        cv2.imwrite(nd_path + '/soggetto_' + str(index) + '_dx.bmp', rotated)

        # Immagine Sinistra

        img = cv2.imread(path + '/soggetto_' + str(index) + '_sx.bmp')


        rotated = imutils.rotate_bound(img, 45)

        rot_dim = rotated.shape[0]
        bordo_a = int((rot_dim - dim_finale) / 2)
        bordo_b = rot_dim - dim_finale - bordo_a

        rotated = rotated[bordo_a:-bordo_b, bordo_a:-bordo_b, :]


        cv2.imwrite(nd_path + '/soggetto_' + str(index) + '_sx.bmp', rotated)

        # Salva csv
        with open((nd_path + '/soggetto_' + str(index) + '.csv'), "w") as f:
            writer = csv.writer(f)
            writer.writerow(data)