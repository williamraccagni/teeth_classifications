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
    max_ages_stages_dict = {}
    min_ages_stages_dict = {}

    for index in range(1, (numero_soggetti + 1) ):

        # Open csv
        with open((path + '/soggetto_' + str(index) + '.csv'), newline='') as f:
            reader = csv.reader(f)
            data = list(reader)[0]



        if data[1] == 'dente presente':

            counter = counter + 1

            if data[4] == 'dente presente':

                if str(data[2]) != 'sconosciuto' and str(data[5]) == 'sconosciuto':
                    # MAX

                    if str(labels[(index - 1)]) in max_ages_stages_dict.keys():
                        if str(data[2]) in max_ages_stages_dict[str(labels[(index - 1)])].keys():
                            max_ages_stages_dict[str(labels[(index - 1)])][str(data[2])] = \
                                max_ages_stages_dict[str(labels[(index - 1)])][str(data[2])] + 1
                        else:
                            max_ages_stages_dict[str(labels[(index - 1)])][str(data[2])] = 1
                    else:
                        max_ages_stages_dict[str(labels[(index - 1)])] = {}
                        max_ages_stages_dict[str(labels[(index - 1)])][str(data[2])] = 1

                    # MIN

                    if str(labels[(index - 1)]) in min_ages_stages_dict.keys():
                        if str(data[2]) in min_ages_stages_dict[str(labels[(index - 1)])].keys():
                            min_ages_stages_dict[str(labels[(index - 1)])][str(data[2])] = \
                                min_ages_stages_dict[str(labels[(index - 1)])][str(data[2])] + 1
                        else:
                            min_ages_stages_dict[str(labels[(index - 1)])][str(data[2])] = 1
                    else:
                        min_ages_stages_dict[str(labels[(index - 1)])] = {}
                        min_ages_stages_dict[str(labels[(index - 1)])][str(data[2])] = 1

                else:
                    if str(data[2]) == 'sconosciuto' and str(data[5]) != 'sconosciuto':
                        # MAX

                        if str(labels[(index - 1)]) in max_ages_stages_dict.keys():
                            if str(data[5]) in max_ages_stages_dict[str(labels[(index - 1)])].keys():
                                max_ages_stages_dict[str(labels[(index - 1)])][str(data[5])] = \
                                    max_ages_stages_dict[str(labels[(index - 1)])][str(data[5])] + 1
                            else:
                                max_ages_stages_dict[str(labels[(index - 1)])][str(data[5])] = 1
                        else:
                            max_ages_stages_dict[str(labels[(index - 1)])] = {}
                            max_ages_stages_dict[str(labels[(index - 1)])][str(data[5])] = 1

                        # MIN

                        if str(labels[(index - 1)]) in min_ages_stages_dict.keys():
                            if str(data[5]) in min_ages_stages_dict[str(labels[(index - 1)])].keys():
                                min_ages_stages_dict[str(labels[(index - 1)])][str(data[5])] = \
                                    min_ages_stages_dict[str(labels[(index - 1)])][str(data[5])] + 1
                            else:
                                min_ages_stages_dict[str(labels[(index - 1)])][str(data[5])] = 1
                        else:
                            min_ages_stages_dict[str(labels[(index - 1)])] = {}
                            min_ages_stages_dict[str(labels[(index - 1)])][str(data[5])] = 1
                    else:

                        if stage_comparison(str(data[2]), str(data[5])) == 1:
                            # MAX

                            if str(labels[(index - 1)]) in max_ages_stages_dict.keys():
                                if str(data[2]) in max_ages_stages_dict[str(labels[(index - 1)])].keys():
                                    max_ages_stages_dict[str(labels[(index - 1)])][str(data[2])] = \
                                        max_ages_stages_dict[str(labels[(index - 1)])][str(data[2])] + 1
                                else:
                                    max_ages_stages_dict[str(labels[(index - 1)])][str(data[2])] = 1
                            else:
                                max_ages_stages_dict[str(labels[(index - 1)])] = {}
                                max_ages_stages_dict[str(labels[(index - 1)])][str(data[2])] = 1

                            # MIN

                            if str(labels[(index - 1)]) in min_ages_stages_dict.keys():
                                if str(data[5]) in min_ages_stages_dict[str(labels[(index - 1)])].keys():
                                    min_ages_stages_dict[str(labels[(index - 1)])][str(data[5])] = \
                                        min_ages_stages_dict[str(labels[(index - 1)])][str(data[5])] + 1
                                else:
                                    min_ages_stages_dict[str(labels[(index - 1)])][str(data[5])] = 1
                            else:
                                min_ages_stages_dict[str(labels[(index - 1)])] = {}
                                min_ages_stages_dict[str(labels[(index - 1)])][str(data[5])] = 1


                        else:
                            if stage_comparison(str(data[2]), str(data[5])) == -1:

                                # MAX

                                if str(labels[(index - 1)]) in max_ages_stages_dict.keys():
                                    if str(data[5]) in max_ages_stages_dict[str(labels[(index - 1)])].keys():
                                        max_ages_stages_dict[str(labels[(index - 1)])][str(data[5])] = \
                                            max_ages_stages_dict[str(labels[(index - 1)])][str(data[5])] + 1
                                    else:
                                        max_ages_stages_dict[str(labels[(index - 1)])][str(data[5])] = 1
                                else:
                                    max_ages_stages_dict[str(labels[(index - 1)])] = {}
                                    max_ages_stages_dict[str(labels[(index - 1)])][str(data[5])] = 1

                                # MIN

                                if str(labels[(index - 1)]) in min_ages_stages_dict.keys():
                                    if str(data[2]) in min_ages_stages_dict[str(labels[(index - 1)])].keys():
                                        min_ages_stages_dict[str(labels[(index - 1)])][str(data[2])] = \
                                            min_ages_stages_dict[str(labels[(index - 1)])][str(data[2])] + 1
                                    else:
                                        min_ages_stages_dict[str(labels[(index - 1)])][str(data[2])] = 1
                                else:
                                    min_ages_stages_dict[str(labels[(index - 1)])] = {}
                                    min_ages_stages_dict[str(labels[(index - 1)])][str(data[2])] = 1


                            else:

                                # MAX

                                if str(labels[(index - 1)]) in max_ages_stages_dict.keys():
                                    if str(data[2]) in max_ages_stages_dict[str(labels[(index - 1)])].keys():
                                        max_ages_stages_dict[str(labels[(index - 1)])][str(data[2])] = \
                                            max_ages_stages_dict[str(labels[(index - 1)])][str(data[2])] + 1
                                    else:
                                        max_ages_stages_dict[str(labels[(index - 1)])][str(data[2])] = 1
                                else:
                                    max_ages_stages_dict[str(labels[(index - 1)])] = {}
                                    max_ages_stages_dict[str(labels[(index - 1)])][str(data[2])] = 1

                                # MIN

                                if str(labels[(index - 1)]) in min_ages_stages_dict.keys():
                                    if str(data[2]) in min_ages_stages_dict[str(labels[(index - 1)])].keys():
                                        min_ages_stages_dict[str(labels[(index - 1)])][str(data[2])] = \
                                            min_ages_stages_dict[str(labels[(index - 1)])][str(data[2])] + 1
                                    else:
                                        min_ages_stages_dict[str(labels[(index - 1)])][str(data[2])] = 1
                                else:
                                    min_ages_stages_dict[str(labels[(index - 1)])] = {}
                                    min_ages_stages_dict[str(labels[(index - 1)])][str(data[2])] = 1

            else:

                # MAX

                if str(labels[(index - 1)]) in max_ages_stages_dict.keys():
                    if str(data[2]) in max_ages_stages_dict[str(labels[(index - 1)])].keys():
                        max_ages_stages_dict[str(labels[(index - 1)])][str(data[2])] = \
                        max_ages_stages_dict[str(labels[(index - 1)])][str(data[2])] + 1
                    else:
                        max_ages_stages_dict[str(labels[(index - 1)])][str(data[2])] = 1
                else:
                    max_ages_stages_dict[str(labels[(index - 1)])] = {}
                    max_ages_stages_dict[str(labels[(index - 1)])][str(data[2])] = 1

                # MIN

                if str(labels[(index - 1)]) in min_ages_stages_dict.keys():
                    if str(data[2]) in min_ages_stages_dict[str(labels[(index - 1)])].keys():
                        min_ages_stages_dict[str(labels[(index - 1)])][str(data[2])] = \
                        min_ages_stages_dict[str(labels[(index - 1)])][str(data[2])] + 1
                    else:
                        min_ages_stages_dict[str(labels[(index - 1)])][str(data[2])] = 1
                else:
                    min_ages_stages_dict[str(labels[(index - 1)])] = {}
                    min_ages_stages_dict[str(labels[(index - 1)])][str(data[2])] = 1

        else:
            if data[4] == 'dente presente':

                counter = counter + 1

                # MAX

                if str(labels[(index - 1)]) in max_ages_stages_dict.keys():
                    if str(data[5]) in max_ages_stages_dict[str(labels[(index - 1)])].keys():
                        max_ages_stages_dict[str(labels[(index - 1)])][str(data[5])] = \
                            max_ages_stages_dict[str(labels[(index - 1)])][str(data[5])] + 1
                    else:
                        max_ages_stages_dict[str(labels[(index - 1)])][str(data[5])] = 1
                else:
                    max_ages_stages_dict[str(labels[(index - 1)])] = {}
                    max_ages_stages_dict[str(labels[(index - 1)])][str(data[5])] = 1

                # MIN

                if str(labels[(index - 1)]) in min_ages_stages_dict.keys():
                    if str(data[5]) in min_ages_stages_dict[str(labels[(index - 1)])].keys():
                        min_ages_stages_dict[str(labels[(index - 1)])][str(data[5])] = \
                            min_ages_stages_dict[str(labels[(index - 1)])][str(data[5])] + 1
                    else:
                        min_ages_stages_dict[str(labels[(index - 1)])][str(data[5])] = 1
                else:
                    min_ages_stages_dict[str(labels[(index - 1)])] = {}
                    min_ages_stages_dict[str(labels[(index - 1)])][str(data[5])] = 1



    print('numero soggetti:')
    print(counter)
    print('MAX STAGES:')
    print(max_ages_stages_dict)
    print('MIN STAGES:')
    print(min_ages_stages_dict)




    print('\n')

    # Maggiorenne minorenne stage

    # Load labels
    labels = magmin_ch_labels()

    print(path)
    print(numero_soggetti)
    print(len(labels))
    print(labels)

    counter = 0
    max_ages_stages_dict = {}
    min_ages_stages_dict = {}

    for index in range(1, (numero_soggetti + 1)):

        # Open csv
        with open((path + '/soggetto_' + str(index) + '.csv'), newline='') as f:
            reader = csv.reader(f)
            data = list(reader)[0]

        if data[1] == 'dente presente':

            counter = counter + 1

            if data[4] == 'dente presente':

                if str(data[2]) != 'sconosciuto' and str(data[5]) == 'sconosciuto':
                    # MAX

                    if str(labels[(index - 1)]) in max_ages_stages_dict.keys():
                        if str(data[2]) in max_ages_stages_dict[str(labels[(index - 1)])].keys():
                            max_ages_stages_dict[str(labels[(index - 1)])][str(data[2])] = \
                                max_ages_stages_dict[str(labels[(index - 1)])][str(data[2])] + 1
                        else:
                            max_ages_stages_dict[str(labels[(index - 1)])][str(data[2])] = 1
                    else:
                        max_ages_stages_dict[str(labels[(index - 1)])] = {}
                        max_ages_stages_dict[str(labels[(index - 1)])][str(data[2])] = 1

                    # MIN

                    if str(labels[(index - 1)]) in min_ages_stages_dict.keys():
                        if str(data[2]) in min_ages_stages_dict[str(labels[(index - 1)])].keys():
                            min_ages_stages_dict[str(labels[(index - 1)])][str(data[2])] = \
                                min_ages_stages_dict[str(labels[(index - 1)])][str(data[2])] + 1
                        else:
                            min_ages_stages_dict[str(labels[(index - 1)])][str(data[2])] = 1
                    else:
                        min_ages_stages_dict[str(labels[(index - 1)])] = {}
                        min_ages_stages_dict[str(labels[(index - 1)])][str(data[2])] = 1

                else:
                    if str(data[2]) == 'sconosciuto' and str(data[5]) != 'sconosciuto':
                        # MAX

                        if str(labels[(index - 1)]) in max_ages_stages_dict.keys():
                            if str(data[5]) in max_ages_stages_dict[str(labels[(index - 1)])].keys():
                                max_ages_stages_dict[str(labels[(index - 1)])][str(data[5])] = \
                                    max_ages_stages_dict[str(labels[(index - 1)])][str(data[5])] + 1
                            else:
                                max_ages_stages_dict[str(labels[(index - 1)])][str(data[5])] = 1
                        else:
                            max_ages_stages_dict[str(labels[(index - 1)])] = {}
                            max_ages_stages_dict[str(labels[(index - 1)])][str(data[5])] = 1

                        # MIN

                        if str(labels[(index - 1)]) in min_ages_stages_dict.keys():
                            if str(data[5]) in min_ages_stages_dict[str(labels[(index - 1)])].keys():
                                min_ages_stages_dict[str(labels[(index - 1)])][str(data[5])] = \
                                    min_ages_stages_dict[str(labels[(index - 1)])][str(data[5])] + 1
                            else:
                                min_ages_stages_dict[str(labels[(index - 1)])][str(data[5])] = 1
                        else:
                            min_ages_stages_dict[str(labels[(index - 1)])] = {}
                            min_ages_stages_dict[str(labels[(index - 1)])][str(data[5])] = 1
                    else:

                        # if stage maggiore, se maggiore metto data 2 max e l altro in min e viceversa
                        if stage_comparison(str(data[2]), str(data[5])) == 1:
                            # MAX

                            if str(labels[(index - 1)]) in max_ages_stages_dict.keys():
                                if str(data[2]) in max_ages_stages_dict[str(labels[(index - 1)])].keys():
                                    max_ages_stages_dict[str(labels[(index - 1)])][str(data[2])] = \
                                        max_ages_stages_dict[str(labels[(index - 1)])][str(data[2])] + 1
                                else:
                                    max_ages_stages_dict[str(labels[(index - 1)])][str(data[2])] = 1
                            else:
                                max_ages_stages_dict[str(labels[(index - 1)])] = {}
                                max_ages_stages_dict[str(labels[(index - 1)])][str(data[2])] = 1

                            # MIN

                            if str(labels[(index - 1)]) in min_ages_stages_dict.keys():
                                if str(data[5]) in min_ages_stages_dict[str(labels[(index - 1)])].keys():
                                    min_ages_stages_dict[str(labels[(index - 1)])][str(data[5])] = \
                                        min_ages_stages_dict[str(labels[(index - 1)])][str(data[5])] + 1
                                else:
                                    min_ages_stages_dict[str(labels[(index - 1)])][str(data[5])] = 1
                            else:
                                min_ages_stages_dict[str(labels[(index - 1)])] = {}
                                min_ages_stages_dict[str(labels[(index - 1)])][str(data[5])] = 1


                        else:
                            if stage_comparison(str(data[2]), str(data[5])) == -1:

                                # MAX

                                if str(labels[(index - 1)]) in max_ages_stages_dict.keys():
                                    if str(data[5]) in max_ages_stages_dict[str(labels[(index - 1)])].keys():
                                        max_ages_stages_dict[str(labels[(index - 1)])][str(data[5])] = \
                                            max_ages_stages_dict[str(labels[(index - 1)])][str(data[5])] + 1
                                    else:
                                        max_ages_stages_dict[str(labels[(index - 1)])][str(data[5])] = 1
                                else:
                                    max_ages_stages_dict[str(labels[(index - 1)])] = {}
                                    max_ages_stages_dict[str(labels[(index - 1)])][str(data[5])] = 1

                                # MIN

                                if str(labels[(index - 1)]) in min_ages_stages_dict.keys():
                                    if str(data[2]) in min_ages_stages_dict[str(labels[(index - 1)])].keys():
                                        min_ages_stages_dict[str(labels[(index - 1)])][str(data[2])] = \
                                            min_ages_stages_dict[str(labels[(index - 1)])][str(data[2])] + 1
                                    else:
                                        min_ages_stages_dict[str(labels[(index - 1)])][str(data[2])] = 1
                                else:
                                    min_ages_stages_dict[str(labels[(index - 1)])] = {}
                                    min_ages_stages_dict[str(labels[(index - 1)])][str(data[2])] = 1


                            else:

                                # MAX

                                if str(labels[(index - 1)]) in max_ages_stages_dict.keys():
                                    if str(data[2]) in max_ages_stages_dict[str(labels[(index - 1)])].keys():
                                        max_ages_stages_dict[str(labels[(index - 1)])][str(data[2])] = \
                                            max_ages_stages_dict[str(labels[(index - 1)])][str(data[2])] + 1
                                    else:
                                        max_ages_stages_dict[str(labels[(index - 1)])][str(data[2])] = 1
                                else:
                                    max_ages_stages_dict[str(labels[(index - 1)])] = {}
                                    max_ages_stages_dict[str(labels[(index - 1)])][str(data[2])] = 1

                                # MIN

                                if str(labels[(index - 1)]) in min_ages_stages_dict.keys():
                                    if str(data[2]) in min_ages_stages_dict[str(labels[(index - 1)])].keys():
                                        min_ages_stages_dict[str(labels[(index - 1)])][str(data[2])] = \
                                            min_ages_stages_dict[str(labels[(index - 1)])][str(data[2])] + 1
                                    else:
                                        min_ages_stages_dict[str(labels[(index - 1)])][str(data[2])] = 1
                                else:
                                    min_ages_stages_dict[str(labels[(index - 1)])] = {}
                                    min_ages_stages_dict[str(labels[(index - 1)])][str(data[2])] = 1

            else:

                # MAX

                if str(labels[(index - 1)]) in max_ages_stages_dict.keys():
                    if str(data[2]) in max_ages_stages_dict[str(labels[(index - 1)])].keys():
                        max_ages_stages_dict[str(labels[(index - 1)])][str(data[2])] = \
                            max_ages_stages_dict[str(labels[(index - 1)])][str(data[2])] + 1
                    else:
                        max_ages_stages_dict[str(labels[(index - 1)])][str(data[2])] = 1
                else:
                    max_ages_stages_dict[str(labels[(index - 1)])] = {}
                    max_ages_stages_dict[str(labels[(index - 1)])][str(data[2])] = 1

                # MIN

                if str(labels[(index - 1)]) in min_ages_stages_dict.keys():
                    if str(data[2]) in min_ages_stages_dict[str(labels[(index - 1)])].keys():
                        min_ages_stages_dict[str(labels[(index - 1)])][str(data[2])] = \
                            min_ages_stages_dict[str(labels[(index - 1)])][str(data[2])] + 1
                    else:
                        min_ages_stages_dict[str(labels[(index - 1)])][str(data[2])] = 1
                else:
                    min_ages_stages_dict[str(labels[(index - 1)])] = {}
                    min_ages_stages_dict[str(labels[(index - 1)])][str(data[2])] = 1

        else:
            if data[4] == 'dente presente':

                counter = counter + 1

                # MAX

                if str(labels[(index - 1)]) in max_ages_stages_dict.keys():
                    if str(data[5]) in max_ages_stages_dict[str(labels[(index - 1)])].keys():
                        max_ages_stages_dict[str(labels[(index - 1)])][str(data[5])] = \
                            max_ages_stages_dict[str(labels[(index - 1)])][str(data[5])] + 1
                    else:
                        max_ages_stages_dict[str(labels[(index - 1)])][str(data[5])] = 1
                else:
                    max_ages_stages_dict[str(labels[(index - 1)])] = {}
                    max_ages_stages_dict[str(labels[(index - 1)])][str(data[5])] = 1

                # MIN

                if str(labels[(index - 1)]) in min_ages_stages_dict.keys():
                    if str(data[5]) in min_ages_stages_dict[str(labels[(index - 1)])].keys():
                        min_ages_stages_dict[str(labels[(index - 1)])][str(data[5])] = \
                            min_ages_stages_dict[str(labels[(index - 1)])][str(data[5])] + 1
                    else:
                        min_ages_stages_dict[str(labels[(index - 1)])][str(data[5])] = 1
                else:
                    min_ages_stages_dict[str(labels[(index - 1)])] = {}
                    min_ages_stages_dict[str(labels[(index - 1)])][str(data[5])] = 1

    print('numero soggetti:')
    print(counter)
    print('MAX STAGES:')
    print(max_ages_stages_dict)
    print('MIN STAGES:')
    print(min_ages_stages_dict)
