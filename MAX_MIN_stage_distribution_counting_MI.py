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
    path = './stadiazione_rx_opt_MI'
    numero_soggetti = 411

    # labels = []

    print(path)
    print(numero_soggetti)

    counter = 0
    max_ages_stages_dict = {}
    min_ages_stages_dict = {}

    for index in range(1, (numero_soggetti + 1) ):

        # Open csv
        with open((path + '/soggetto_' + str(index) + '.csv'), newline='') as f:
            reader = csv.reader(f)
            data = list(reader)[0]

        cur_label = label_converter( int(data[9]), int(data[10]))
        if(cur_label == 'error'):
            print('ERRORE')



        if data[1] == 'dente presente':

            counter = counter + 1

            if data[4] == 'dente presente':

                if str(data[2]) != 'sconosciuto' and str(data[5]) == 'sconosciuto':

                    # MAX

                    if str(cur_label) in max_ages_stages_dict.keys():
                        if str(data[2]) in max_ages_stages_dict[str(cur_label)].keys():
                            max_ages_stages_dict[str(cur_label)][str(data[2])] = \
                                max_ages_stages_dict[str(cur_label)][str(data[2])] + 1
                        else:
                            max_ages_stages_dict[str(cur_label)][str(data[2])] = 1
                    else:
                        max_ages_stages_dict[str(cur_label)] = {}
                        max_ages_stages_dict[str(cur_label)][str(data[2])] = 1

                    # MIN

                    if str(cur_label) in min_ages_stages_dict.keys():
                        if str(data[2]) in min_ages_stages_dict[str(cur_label)].keys():
                            min_ages_stages_dict[str(cur_label)][str(data[2])] = \
                                min_ages_stages_dict[str(cur_label)][str(data[2])] + 1
                        else:
                            min_ages_stages_dict[str(cur_label)][str(data[2])] = 1
                    else:
                        min_ages_stages_dict[str(cur_label)] = {}
                        min_ages_stages_dict[str(cur_label)][str(data[2])] = 1


                else:

                    if str(data[2]) == 'sconosciuto' and str(data[5]) != 'sconosciuto':

                        # MAX

                        if str(cur_label) in max_ages_stages_dict.keys():
                            if str(data[5]) in max_ages_stages_dict[str(cur_label)].keys():
                                max_ages_stages_dict[str(cur_label)][str(data[5])] = \
                                    max_ages_stages_dict[str(cur_label)][str(data[5])] + 1
                            else:
                                max_ages_stages_dict[str(cur_label)][str(data[5])] = 1
                        else:
                            max_ages_stages_dict[str(cur_label)] = {}
                            max_ages_stages_dict[str(cur_label)][str(data[5])] = 1

                        # MIN

                        if str(cur_label) in min_ages_stages_dict.keys():
                            if str(data[5]) in min_ages_stages_dict[str(cur_label)].keys():
                                min_ages_stages_dict[str(cur_label)][str(data[5])] = \
                                    min_ages_stages_dict[str(cur_label)][str(data[5])] + 1
                            else:
                                min_ages_stages_dict[str(cur_label)][str(data[5])] = 1
                        else:
                            min_ages_stages_dict[str(cur_label)] = {}
                            min_ages_stages_dict[str(cur_label)][str(data[5])] = 1

                    else:

                        # if stage maggiore, se maggiore metto data 2 max e l altro in min e viceversa
                        if stage_comparison(str(data[2]), str(data[5])) == 1:
                            # MAX

                            if str(cur_label) in max_ages_stages_dict.keys():
                                if str(data[2]) in max_ages_stages_dict[str(cur_label)].keys():
                                    max_ages_stages_dict[str(cur_label)][str(data[2])] = \
                                        max_ages_stages_dict[str(cur_label)][str(data[2])] + 1
                                else:
                                    max_ages_stages_dict[str(cur_label)][str(data[2])] = 1
                            else:
                                max_ages_stages_dict[str(cur_label)] = {}
                                max_ages_stages_dict[str(cur_label)][str(data[2])] = 1

                            # MIN

                            if str(cur_label) in min_ages_stages_dict.keys():
                                if str(data[5]) in min_ages_stages_dict[str(cur_label)].keys():
                                    min_ages_stages_dict[str(cur_label)][str(data[5])] = \
                                        min_ages_stages_dict[str(cur_label)][str(data[5])] + 1
                                else:
                                    min_ages_stages_dict[str(cur_label)][str(data[5])] = 1
                            else:
                                min_ages_stages_dict[str(cur_label)] = {}
                                min_ages_stages_dict[str(cur_label)][str(data[5])] = 1


                        else:
                            if stage_comparison(str(data[2]), str(data[5])) == -1:

                                # MAX

                                if str(cur_label) in max_ages_stages_dict.keys():
                                    if str(data[5]) in max_ages_stages_dict[str(cur_label)].keys():
                                        max_ages_stages_dict[str(cur_label)][str(data[5])] = \
                                            max_ages_stages_dict[str(cur_label)][str(data[5])] + 1
                                    else:
                                        max_ages_stages_dict[str(cur_label)][str(data[5])] = 1
                                else:
                                    max_ages_stages_dict[str(cur_label)] = {}
                                    max_ages_stages_dict[str(cur_label)][str(data[5])] = 1

                                # MIN

                                if str(cur_label) in min_ages_stages_dict.keys():
                                    if str(data[2]) in min_ages_stages_dict[str(cur_label)].keys():
                                        min_ages_stages_dict[str(cur_label)][str(data[2])] = \
                                            min_ages_stages_dict[str(cur_label)][str(data[2])] + 1
                                    else:
                                        min_ages_stages_dict[str(cur_label)][str(data[2])] = 1
                                else:
                                    min_ages_stages_dict[str(cur_label)] = {}
                                    min_ages_stages_dict[str(cur_label)][str(data[2])] = 1


                            else:

                                # MAX

                                if str(cur_label) in max_ages_stages_dict.keys():
                                    if str(data[2]) in max_ages_stages_dict[str(cur_label)].keys():
                                        max_ages_stages_dict[str(cur_label)][str(data[2])] = \
                                            max_ages_stages_dict[str(cur_label)][str(data[2])] + 1
                                    else:
                                        max_ages_stages_dict[str(cur_label)][str(data[2])] = 1
                                else:
                                    max_ages_stages_dict[str(cur_label)] = {}
                                    max_ages_stages_dict[str(cur_label)][str(data[2])] = 1

                                # MIN

                                if str(cur_label) in min_ages_stages_dict.keys():
                                    if str(data[2]) in min_ages_stages_dict[str(cur_label)].keys():
                                        min_ages_stages_dict[str(cur_label)][str(data[2])] = \
                                            min_ages_stages_dict[str(cur_label)][str(data[2])] + 1
                                    else:
                                        min_ages_stages_dict[str(cur_label)][str(data[2])] = 1
                                else:
                                    min_ages_stages_dict[str(cur_label)] = {}
                                    min_ages_stages_dict[str(cur_label)][str(data[2])] = 1

            else:

                # MAX

                if str(cur_label) in max_ages_stages_dict.keys():
                    if str(data[2]) in max_ages_stages_dict[str(cur_label)].keys():
                        max_ages_stages_dict[str(cur_label)][str(data[2])] = \
                        max_ages_stages_dict[str(cur_label)][str(data[2])] + 1
                    else:
                        max_ages_stages_dict[str(cur_label)][str(data[2])] = 1
                else:
                    max_ages_stages_dict[str(cur_label)] = {}
                    max_ages_stages_dict[str(cur_label)][str(data[2])] = 1

                # MIN

                if str(cur_label) in min_ages_stages_dict.keys():
                    if str(data[2]) in min_ages_stages_dict[str(cur_label)].keys():
                        min_ages_stages_dict[str(cur_label)][str(data[2])] = \
                        min_ages_stages_dict[str(cur_label)][str(data[2])] + 1
                    else:
                        min_ages_stages_dict[str(cur_label)][str(data[2])] = 1
                else:
                    min_ages_stages_dict[str(cur_label)] = {}
                    min_ages_stages_dict[str(cur_label)][str(data[2])] = 1

        else:
            if data[4] == 'dente presente':

                counter = counter + 1

                # MAX

                if str(cur_label) in max_ages_stages_dict.keys():
                    if str(data[5]) in max_ages_stages_dict[str(cur_label)].keys():
                        max_ages_stages_dict[str(cur_label)][str(data[5])] = \
                            max_ages_stages_dict[str(cur_label)][str(data[5])] + 1
                    else:
                        max_ages_stages_dict[str(cur_label)][str(data[5])] = 1
                else:
                    max_ages_stages_dict[str(cur_label)] = {}
                    max_ages_stages_dict[str(cur_label)][str(data[5])] = 1

                # MIN

                if str(cur_label) in min_ages_stages_dict.keys():
                    if str(data[5]) in min_ages_stages_dict[str(cur_label)].keys():
                        min_ages_stages_dict[str(cur_label)][str(data[5])] = \
                            min_ages_stages_dict[str(cur_label)][str(data[5])] + 1
                    else:
                        min_ages_stages_dict[str(cur_label)][str(data[5])] = 1
                else:
                    min_ages_stages_dict[str(cur_label)] = {}
                    min_ages_stages_dict[str(cur_label)][str(data[5])] = 1



    print('numero soggetti:')
    print(counter)
    print('MAX STAGES:')
    print(max_ages_stages_dict)
    print('MIN STAGES:')
    print(min_ages_stages_dict)





    print('\n')

    # Maggiorenne minorenne stage

    print(path)
    print(numero_soggetti)

    counter = 0
    max_ages_stages_dict = {}
    min_ages_stages_dict = {}

    for index in range(1, (numero_soggetti + 1) ):

        # Open csv
        with open((path + '/soggetto_' + str(index) + '.csv'), newline='') as f:
            reader = csv.reader(f)
            data = list(reader)[0]

        cur_label = magmin_label( int(data[9]), int(data[10]))
        if(cur_label == 'error'):
            print('ERRORE')

        if data[1] == 'dente presente':

            counter = counter + 1

            if data[4] == 'dente presente':

                if str(data[2]) != 'sconosciuto' and str(data[5]) == 'sconosciuto':

                    # MAX

                    if str(cur_label) in max_ages_stages_dict.keys():
                        if str(data[2]) in max_ages_stages_dict[str(cur_label)].keys():
                            max_ages_stages_dict[str(cur_label)][str(data[2])] = \
                                max_ages_stages_dict[str(cur_label)][str(data[2])] + 1
                        else:
                            max_ages_stages_dict[str(cur_label)][str(data[2])] = 1
                    else:
                        max_ages_stages_dict[str(cur_label)] = {}
                        max_ages_stages_dict[str(cur_label)][str(data[2])] = 1

                    # MIN

                    if str(cur_label) in min_ages_stages_dict.keys():
                        if str(data[2]) in min_ages_stages_dict[str(cur_label)].keys():
                            min_ages_stages_dict[str(cur_label)][str(data[2])] = \
                                min_ages_stages_dict[str(cur_label)][str(data[2])] + 1
                        else:
                            min_ages_stages_dict[str(cur_label)][str(data[2])] = 1
                    else:
                        min_ages_stages_dict[str(cur_label)] = {}
                        min_ages_stages_dict[str(cur_label)][str(data[2])] = 1


                else:

                    if str(data[2]) == 'sconosciuto' and str(data[5]) != 'sconosciuto':

                        # MAX

                        if str(cur_label) in max_ages_stages_dict.keys():
                            if str(data[5]) in max_ages_stages_dict[str(cur_label)].keys():
                                max_ages_stages_dict[str(cur_label)][str(data[5])] = \
                                    max_ages_stages_dict[str(cur_label)][str(data[5])] + 1
                            else:
                                max_ages_stages_dict[str(cur_label)][str(data[5])] = 1
                        else:
                            max_ages_stages_dict[str(cur_label)] = {}
                            max_ages_stages_dict[str(cur_label)][str(data[5])] = 1

                        # MIN

                        if str(cur_label) in min_ages_stages_dict.keys():
                            if str(data[5]) in min_ages_stages_dict[str(cur_label)].keys():
                                min_ages_stages_dict[str(cur_label)][str(data[5])] = \
                                    min_ages_stages_dict[str(cur_label)][str(data[5])] + 1
                            else:
                                min_ages_stages_dict[str(cur_label)][str(data[5])] = 1
                        else:
                            min_ages_stages_dict[str(cur_label)] = {}
                            min_ages_stages_dict[str(cur_label)][str(data[5])] = 1

                    else:

                        # if stage maggiore, se maggiore metto data 2 max e l altro in min e viceversa
                        if stage_comparison(str(data[2]), str(data[5])) == 1:
                            # MAX

                            if str(cur_label) in max_ages_stages_dict.keys():
                                if str(data[2]) in max_ages_stages_dict[str(cur_label)].keys():
                                    max_ages_stages_dict[str(cur_label)][str(data[2])] = \
                                        max_ages_stages_dict[str(cur_label)][str(data[2])] + 1
                                else:
                                    max_ages_stages_dict[str(cur_label)][str(data[2])] = 1
                            else:
                                max_ages_stages_dict[str(cur_label)] = {}
                                max_ages_stages_dict[str(cur_label)][str(data[2])] = 1

                            # MIN

                            if str(cur_label) in min_ages_stages_dict.keys():
                                if str(data[5]) in min_ages_stages_dict[str(cur_label)].keys():
                                    min_ages_stages_dict[str(cur_label)][str(data[5])] = \
                                        min_ages_stages_dict[str(cur_label)][str(data[5])] + 1
                                else:
                                    min_ages_stages_dict[str(cur_label)][str(data[5])] = 1
                            else:
                                min_ages_stages_dict[str(cur_label)] = {}
                                min_ages_stages_dict[str(cur_label)][str(data[5])] = 1


                        else:
                            if stage_comparison(str(data[2]), str(data[5])) == -1:

                                # MAX

                                if str(cur_label) in max_ages_stages_dict.keys():
                                    if str(data[5]) in max_ages_stages_dict[str(cur_label)].keys():
                                        max_ages_stages_dict[str(cur_label)][str(data[5])] = \
                                            max_ages_stages_dict[str(cur_label)][str(data[5])] + 1
                                    else:
                                        max_ages_stages_dict[str(cur_label)][str(data[5])] = 1
                                else:
                                    max_ages_stages_dict[str(cur_label)] = {}
                                    max_ages_stages_dict[str(cur_label)][str(data[5])] = 1

                                # MIN

                                if str(cur_label) in min_ages_stages_dict.keys():
                                    if str(data[2]) in min_ages_stages_dict[str(cur_label)].keys():
                                        min_ages_stages_dict[str(cur_label)][str(data[2])] = \
                                            min_ages_stages_dict[str(cur_label)][str(data[2])] + 1
                                    else:
                                        min_ages_stages_dict[str(cur_label)][str(data[2])] = 1
                                else:
                                    min_ages_stages_dict[str(cur_label)] = {}
                                    min_ages_stages_dict[str(cur_label)][str(data[2])] = 1


                            else:

                                # MAX

                                if str(cur_label) in max_ages_stages_dict.keys():
                                    if str(data[2]) in max_ages_stages_dict[str(cur_label)].keys():
                                        max_ages_stages_dict[str(cur_label)][str(data[2])] = \
                                            max_ages_stages_dict[str(cur_label)][str(data[2])] + 1
                                    else:
                                        max_ages_stages_dict[str(cur_label)][str(data[2])] = 1
                                else:
                                    max_ages_stages_dict[str(cur_label)] = {}
                                    max_ages_stages_dict[str(cur_label)][str(data[2])] = 1

                                # MIN

                                if str(cur_label) in min_ages_stages_dict.keys():
                                    if str(data[2]) in min_ages_stages_dict[str(cur_label)].keys():
                                        min_ages_stages_dict[str(cur_label)][str(data[2])] = \
                                            min_ages_stages_dict[str(cur_label)][str(data[2])] + 1
                                    else:
                                        min_ages_stages_dict[str(cur_label)][str(data[2])] = 1
                                else:
                                    min_ages_stages_dict[str(cur_label)] = {}
                                    min_ages_stages_dict[str(cur_label)][str(data[2])] = 1

            else:

                # MAX

                if str(cur_label) in max_ages_stages_dict.keys():
                    if str(data[2]) in max_ages_stages_dict[str(cur_label)].keys():
                        max_ages_stages_dict[str(cur_label)][str(data[2])] = \
                        max_ages_stages_dict[str(cur_label)][str(data[2])] + 1
                    else:
                        max_ages_stages_dict[str(cur_label)][str(data[2])] = 1
                else:
                    max_ages_stages_dict[str(cur_label)] = {}
                    max_ages_stages_dict[str(cur_label)][str(data[2])] = 1

                # MIN

                if str(cur_label) in min_ages_stages_dict.keys():
                    if str(data[2]) in min_ages_stages_dict[str(cur_label)].keys():
                        min_ages_stages_dict[str(cur_label)][str(data[2])] = \
                        min_ages_stages_dict[str(cur_label)][str(data[2])] + 1
                    else:
                        min_ages_stages_dict[str(cur_label)][str(data[2])] = 1
                else:
                    min_ages_stages_dict[str(cur_label)] = {}
                    min_ages_stages_dict[str(cur_label)][str(data[2])] = 1

        else:
            if data[4] == 'dente presente':

                counter = counter + 1

                # MAX

                if str(cur_label) in max_ages_stages_dict.keys():
                    if str(data[5]) in max_ages_stages_dict[str(cur_label)].keys():
                        max_ages_stages_dict[str(cur_label)][str(data[5])] = \
                            max_ages_stages_dict[str(cur_label)][str(data[5])] + 1
                    else:
                        max_ages_stages_dict[str(cur_label)][str(data[5])] = 1
                else:
                    max_ages_stages_dict[str(cur_label)] = {}
                    max_ages_stages_dict[str(cur_label)][str(data[5])] = 1

                # MIN

                if str(cur_label) in min_ages_stages_dict.keys():
                    if str(data[5]) in min_ages_stages_dict[str(cur_label)].keys():
                        min_ages_stages_dict[str(cur_label)][str(data[5])] = \
                            min_ages_stages_dict[str(cur_label)][str(data[5])] + 1
                    else:
                        min_ages_stages_dict[str(cur_label)][str(data[5])] = 1
                else:
                    min_ages_stages_dict[str(cur_label)] = {}
                    min_ages_stages_dict[str(cur_label)][str(data[5])] = 1



    print('numero soggetti:')
    print(counter)
    print('MAX STAGES:')
    print(max_ages_stages_dict)
    print('MIN STAGES:')
    print(min_ages_stages_dict)
