import csv

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

def magmin_ch_labels():

    labels = []

    path = './stadiazione_rx_opt_CH'

    with open((path + '/anni_CH.csv'), newline='') as f:
        reader = csv.reader(f)
        data = list(reader)

    print(data)

    sog_list = [ [int(data[index][1]), int(data[index][2])] for index in range(len(data))]

    print(sog_list)

    for x in sog_list:
        if x[0] >= 18:
            labels.append('mag')
        else:
            labels.append('min')

    print(len(labels))

    return labels

if __name__ == '__main__':

    # CHIETI

    path = './stadiazione_rx_opt_CH'
    numero_soggetti = 237

    # Load labels
    labels = ch_labels()

    print(path)
    print(numero_soggetti)
    print(len(labels))
    print(labels)

    counter = 0
    ages_stages_dict = {}

    for index in range(1, (numero_soggetti + 1) ):

        # Open csv
        with open((path + '/soggetto_' + str(index) + '.csv'), newline='') as f:
            reader = csv.reader(f)
            data = list(reader)[0]

        # immagine Destra

        if data[1] == 'dente presente':

            counter = counter + 1

            if str(labels[(index - 1)]) in ages_stages_dict.keys():
                if str(data[2]) in ages_stages_dict[str(labels[(index - 1)])].keys():
                    ages_stages_dict[str(labels[(index - 1)])][str(data[2])] = ages_stages_dict[str(labels[(index - 1)])][str(data[2])] + 1
                else:
                    ages_stages_dict[str(labels[(index - 1)])][str(data[2])] = 1
            else:
                ages_stages_dict[str(labels[(index - 1)])] = {}
                ages_stages_dict[str(labels[(index - 1)])][str(data[2])] = 1

        # immagine Sinistra

        if data[4] == 'dente presente':

            counter = counter + 1

            if str(labels[(index - 1)]) in ages_stages_dict.keys():
                if str(data[5]) in ages_stages_dict[str(labels[(index - 1)])].keys():
                    ages_stages_dict[str(labels[(index - 1)])][str(data[5])] = \
                    ages_stages_dict[str(labels[(index - 1)])][str(data[5])] + 1
                else:
                    ages_stages_dict[str(labels[(index - 1)])][str(data[5])] = 1
            else:
                ages_stages_dict[str(labels[(index - 1)])] = {}
                ages_stages_dict[str(labels[(index - 1)])][str(data[5])] = 1



    print(counter)
    print(ages_stages_dict)




    print('\n')

    # Maggiorenne minorenne stage

    # Load labels
    labels = magmin_ch_labels()

    print(path)
    print(numero_soggetti)
    print(len(labels))
    print(labels)

    counter = 0
    ages_stages_dict = {}

    for index in range(1, (numero_soggetti + 1) ):

        # Open csv
        with open((path + '/soggetto_' + str(index) + '.csv'), newline='') as f:
            reader = csv.reader(f)
            data = list(reader)[0]

        # immagine Destra

        if data[1] == 'dente presente':

            counter = counter + 1

            if str(labels[(index - 1)]) in ages_stages_dict.keys():
                if str(data[2]) in ages_stages_dict[str(labels[(index - 1)])].keys():
                    ages_stages_dict[str(labels[(index - 1)])][str(data[2])] = ages_stages_dict[str(labels[(index - 1)])][str(data[2])] + 1
                else:
                    ages_stages_dict[str(labels[(index - 1)])][str(data[2])] = 1
            else:
                ages_stages_dict[str(labels[(index - 1)])] = {}
                ages_stages_dict[str(labels[(index - 1)])][str(data[2])] = 1

        # immagine Sinistra

        if data[4] == 'dente presente':

            counter = counter + 1

            if str(labels[(index - 1)]) in ages_stages_dict.keys():
                if str(data[5]) in ages_stages_dict[str(labels[(index - 1)])].keys():
                    ages_stages_dict[str(labels[(index - 1)])][str(data[5])] = \
                    ages_stages_dict[str(labels[(index - 1)])][str(data[5])] + 1
                else:
                    ages_stages_dict[str(labels[(index - 1)])][str(data[5])] = 1
            else:
                ages_stages_dict[str(labels[(index - 1)])] = {}
                ages_stages_dict[str(labels[(index - 1)])][str(data[5])] = 1

    print(counter)
    print(ages_stages_dict)