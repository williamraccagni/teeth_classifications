import csv
import cv2
import numpy as np

def label_converter(years, months):

    if (years == 13 and months >= 7) or (years == 14 and months <= 6):
        return 14
    if (years == 14 and months >= 7) or (years == 15 and months <= 6):
        return 15
    if (years == 15 and months >= 7) or (years == 16 and months <= 6):
        return 16
    if (years == 16 and months >= 7) or (years == 17 and months <= 6):
        return 17
    if (years == 17 and months >= 7) or (years == 18 and months <= 6):
        return 18
    if (years == 18 and months >= 7) or (years == 19 and months <= 6):
        return 19
    if (years == 19 and months >= 7) or (years == 20 and months <= 6):
        return 20
    if (years == 20 and months >= 7) or (years == 21 and months <= 6):
        return 21
    if (years == 21 and months >= 7) or (years == 22 and months <= 6):
        return 22
    if (years == 22 and months >= 7) or (years == 23 and months <= 6):
        return 23

    return 'error'

if __name__ == '__main__':

    # Preparazione Dataset
    path = './stadiazione_rx_opt_MI'
    numero_soggetti = 411

    nd_path = './dataset_vgg_MI'

    original_labels = []
    labels = []

    print(path)
    print(numero_soggetti)
    print(nd_path)

    cornice = 20
    counter = 0



    for index in range(1, (numero_soggetti + 1) ):

        # Open csv
        with open((path + '/soggetto_' + str(index) + '.csv'), newline='') as f:
            reader = csv.reader(f)
            data = list(reader)[0]

        cur_label = label_converter( int(data[9]), int(data[10]))
        if(cur_label != 'error'):
            original_labels.append(cur_label)
        else:
            print('ERRORE')

        # immagine Destra

        if data[1] == 'dente presente':

            img = cv2.imread(path + '/soggetto_'+str(index)+'_dx.bmp')

            img = img[cornice:-cornice,cornice:-cornice, :]

            img = cv2.resize(img, (224,224), interpolation=cv2.INTER_AREA)

            cv2.imwrite(nd_path + '/'+str(counter)+'.bmp', img)

            counter = counter + 1

            labels.append(cur_label)

            # print(img.shape)




        # immagine Sinistra

        if data[4] == 'dente presente':

            img = cv2.imread(path + '/soggetto_' + str(index) + '_sx.bmp')

            img = img[cornice:-cornice,cornice:-cornice, :]

            img = np.flip(img,axis=1)

            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)

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