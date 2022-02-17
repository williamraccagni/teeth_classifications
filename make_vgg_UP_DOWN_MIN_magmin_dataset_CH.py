import csv
import cv2
import numpy as np

def label_converter_magmin(years, months):

    if years >= 18:
        return 1
    else:
        return 0

    return 'error'

def stage_comparison(stage_a, stage_b):

    if stage_a == 'sconosciuto' and stage_b == 'sconosciuto':
        return 0
    else:
        if stage_a == 'sconosciuto' and stage_b != 'sconosciuto':
            return -1
        else:
            if stage_a != 'sconosciuto' and stage_b == 'sconosciuto':
                return 1
            else:
                if stage_a > stage_b:
                    return 1
                else:
                    if stage_a < stage_b:
                        return -1
                    else:
                        return 0

if __name__ == '__main__':

    # Preparazione Dataset
    path = './UP_DOWN_stadiazione_CH_gimp'
    numero_soggetti = 237

    nd_path = 'dataset_vgg_UP_DOWN_MIN_magmin_CH'

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



        # controllo e inserimento

        if data[1] == 'dente presente' and (str(data[2]) != 'sconosciuto' or str(data[3]) != 'sconosciuto'):

            if data[4] == 'dente presente' and (str(data[5]) != 'sconosciuto' or str(data[6]) != 'sconosciuto'):

                if stage_comparison(str(data[2]), str(data[5])) == 1:# take min

                    img = cv2.imread(path + '/soggetto_' + str(index) + '_sx.bmp')

                    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)

                    cv2.imwrite(nd_path + '/' + str(counter) + '.bmp', img)

                    labels.append(cur_label)

                else:
                    if stage_comparison(str(data[2]), str(data[5])) == -1:

                        img = cv2.imread(path + '/soggetto_' + str(index) + '_dx.bmp')

                        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)

                        cv2.imwrite(nd_path + '/' + str(counter) + '.bmp', img)

                        labels.append(cur_label)

                    else:

                        img = cv2.imread(path + '/soggetto_' + str(index) + '_dx.bmp')

                        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)

                        cv2.imwrite(nd_path + '/' + str(counter) + '.bmp', img)

                        counter = counter + 1

                        labels.append(cur_label)

                        img = cv2.imread(path + '/soggetto_' + str(index) + '_sx.bmp')

                        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)

                        cv2.imwrite(nd_path + '/' + str(counter) + '.bmp', img)

                        labels.append(cur_label)

                counter = counter + 1

        #     else:
        #         pass
        #
        #     counter = counter + 1
        # else:
        #
        #     if data[4] == 'dente presente' and (str(data[5]) != 'sconosciuto' or str(data[6]) != 'sconosciuto'):
        #
        #
        #
        #         counter = counter + 1


    # ETICHETTE

    print(numero_soggetti)
    print(counter)

    print(original_labels)
    print(len(original_labels))
    print(labels)
    print(len(labels))

    with open(nd_path + '/labels.csv', 'w', newline='') as f:

        write = csv.writer(f)
        write.writerow(labels)

