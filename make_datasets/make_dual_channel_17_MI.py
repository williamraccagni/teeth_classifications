# 64 x 64

import csv
import cv2
import numpy as np

# def label_converter_magmin(years, months):
#
#     if years >= 18:
#         return 1
#     else:
#         return 0
#
#     return 'error'

if __name__ == '__main__':

    # Preparazione Dataset
    path = './sezioni_17_gimp'

    nd_path = './dual_channel_MI_17'

    # original_labels = []
    # labels = []

    print(path)
    print(nd_path)

    for index in range(127, (173 + 1) ):

        # Open csv
        with open((path + '/soggetto_' + str(index) + '.csv'), newline='') as f:
            reader = csv.reader(f)
            data = list(reader)[0]

        # cur_label = label_converter_magmin( int(data[9]), int(data[10]))
        # if(cur_label != 'error'):
        #     original_labels.append(cur_label)
        # else:
        #     print('ERRORE')



        # controllo e salvataggio

        if data[1] == 'dente presente' and (str(data[2]) != 'sconosciuto' or str(data[3]) != 'sconosciuto'):

            if data[4] == 'dente presente' and (str(data[5]) != 'sconosciuto' or str(data[6]) != 'sconosciuto'):

                img = cv2.imread(path + '/soggetto_' + str(index) + '_dx.bmp', 0)

                img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)

                cv2.imwrite(nd_path + '/' + str(index) + '_dx.bmp', img)

                img = cv2.imread(path + '/soggetto_' + str(index) + '_sx.bmp', 0)

                img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)

                cv2.imwrite(nd_path + '/' + str(index) + '_sx.bmp', img)

            else:

                img = cv2.imread(path + '/soggetto_' + str(index) + '_dx.bmp', 0)

                img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)

                cv2.imwrite(nd_path + '/' + str(index) + '_dx.bmp', img)

                img = cv2.imread(path + '/soggetto_' + str(index) + '_dx.bmp', 0)

                img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)

                img = np.flip(img, axis=1)

                cv2.imwrite(nd_path + '/' + str(index) + '_sx.bmp', img)

                print(index)

            # counter = counter + 1
            #
            # labels.append(cur_label)

        else:

            if data[4] == 'dente presente' and (str(data[5]) != 'sconosciuto' or str(data[6]) != 'sconosciuto'):
                img = cv2.imread(path + '/soggetto_' + str(index) + '_sx.bmp', 0)

                img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)

                img = np.flip(img, axis=1)

                cv2.imwrite(nd_path + '/' + str(index) + '_dx.bmp', img)

                img = cv2.imread(path + '/soggetto_' + str(index) + '_sx.bmp', 0)

                img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)

                cv2.imwrite(nd_path + '/' + str(index) + '_sx.bmp', img)

                print(index)

                # counter = counter + 1
                #
                # labels.append(cur_label)

            else:

                img = cv2.imread(path + '/soggetto_' + str(index) + '_dx.bmp', 0)

                img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)

                cv2.imwrite(nd_path + '/' + str(index) + '_dx.bmp', img)

                img = cv2.imread(path + '/soggetto_' + str(index) + '_sx.bmp', 0)

                img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)

                cv2.imwrite(nd_path + '/' + str(index) + '_sx.bmp', img)

