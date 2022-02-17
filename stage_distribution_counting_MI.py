import csv

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

def magmin_label(years, months):

    if years >= 18:
        return 'mag'
    else:
        return 'min'

    return 'error'

if __name__ == '__main__':

    # Preparazione Dataset
    path = './stadiazione_rx_opt_MI'
    numero_soggetti = 411

    # labels = []

    print(path)
    print(numero_soggetti)

    counter = 0
    ages_stages_dict = {}

    for index in range(1, (numero_soggetti + 1) ):

        # Open csv
        with open((path + '/soggetto_' + str(index) + '.csv'), newline='') as f:
            reader = csv.reader(f)
            data = list(reader)[0]

        cur_label = label_converter( int(data[9]), int(data[10]))
        if(cur_label == 'error'):
            print('ERRORE')

        # immagine Destra

        if data[1] == 'dente presente':

            counter = counter + 1

            if str(cur_label) in ages_stages_dict.keys():
                if str(data[2]) in ages_stages_dict[str(cur_label)].keys():
                    ages_stages_dict[str(cur_label)][str(data[2])] = \
                    ages_stages_dict[str(cur_label)][str(data[2])] + 1
                else:
                    ages_stages_dict[str(cur_label)][str(data[2])] = 1
            else:
                ages_stages_dict[str(cur_label)] = {}
                ages_stages_dict[str(cur_label)][str(data[2])] = 1


        # immagine Sinistra

        if data[4] == 'dente presente':

            counter = counter + 1

            if str(cur_label) in ages_stages_dict.keys():
                if str(data[5]) in ages_stages_dict[str(cur_label)].keys():
                    ages_stages_dict[str(cur_label)][str(data[5])] = \
                    ages_stages_dict[str(cur_label)][str(data[5])] + 1
                else:
                    ages_stages_dict[str(cur_label)][str(data[5])] = 1
            else:
                ages_stages_dict[str(cur_label)] = {}
                ages_stages_dict[str(cur_label)][str(data[5])] = 1



    print(counter)
    print(ages_stages_dict)





    print('\n')

    # Maggiorenne minorenne stage

    print(path)
    print(numero_soggetti)

    counter = 0
    ages_stages_dict = {}

    for index in range(1, (numero_soggetti + 1) ):

        # Open csv
        with open((path + '/soggetto_' + str(index) + '.csv'), newline='') as f:
            reader = csv.reader(f)
            data = list(reader)[0]

        cur_label = magmin_label( int(data[9]), int(data[10]))
        if(cur_label == 'error'):
            print('ERRORE')

        # immagine Destra

        if data[1] == 'dente presente':

            counter = counter + 1

            if str(cur_label) in ages_stages_dict.keys():
                if str(data[2]) in ages_stages_dict[str(cur_label)].keys():
                    ages_stages_dict[str(cur_label)][str(data[2])] = \
                    ages_stages_dict[str(cur_label)][str(data[2])] + 1
                else:
                    ages_stages_dict[str(cur_label)][str(data[2])] = 1
            else:
                ages_stages_dict[str(cur_label)] = {}
                ages_stages_dict[str(cur_label)][str(data[2])] = 1


        # immagine Sinistra

        if data[4] == 'dente presente':

            counter = counter + 1

            if str(cur_label) in ages_stages_dict.keys():
                if str(data[5]) in ages_stages_dict[str(cur_label)].keys():
                    ages_stages_dict[str(cur_label)][str(data[5])] = \
                    ages_stages_dict[str(cur_label)][str(data[5])] + 1
                else:
                    ages_stages_dict[str(cur_label)][str(data[5])] = 1
            else:
                ages_stages_dict[str(cur_label)] = {}
                ages_stages_dict[str(cur_label)][str(data[5])] = 1



    print(counter)
    print(ages_stages_dict)
