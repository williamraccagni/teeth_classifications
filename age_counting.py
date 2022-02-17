import csv
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # ages = {'15' : 0, '16' : 0, '17' : 0, '18' : 0, '19' : 0, '20' : 0, '21' : 0, '22' : 0}
    #
    # path = './stadiazione_rx_opt_CH'
    #
    # with open((path + '/anni_CH.csv'), newline='') as f:
    #     reader = csv.reader(f)
    #     data = list(reader)
    #
    # print(data)
    #
    # sog_list = [ [int(x[1]), int(x[2])] for x in data]
    #
    # print(sog_list)
    #
    # # min_year = 100
    # # max_year = 0
    # #
    # # for x in sog_list:
    # #     if x[0] < min_year:
    # #         min_year = x[0]
    # #     if x[0] > max_year:
    # #         max_year = x[0]
    # #
    # # print(min_year)
    # # print(max_year)
    #
    # for x in sog_list:
    #     if (x[0] == 14 and x[1] >= 7) or (x[0] == 15 and x[1] <= 6):
    #         ages['15'] = ages['15'] + 1
    #     if (x[0] == 15 and x[1] >= 7) or (x[0] == 16 and x[1] <= 6):
    #         ages['16'] = ages['16'] + 1
    #     if (x[0] == 16 and x[1] >= 7) or (x[0] == 17 and x[1] <= 6):
    #         ages['17'] = ages['17'] + 1
    #     if (x[0] == 17 and x[1] >= 7) or (x[0] == 18 and x[1] <= 6):
    #         ages['18'] = ages['18'] + 1
    #     if (x[0] == 18 and x[1] >= 7) or (x[0] == 19 and x[1] <= 6):
    #         ages['19'] = ages['19'] + 1
    #     if (x[0] == 19 and x[1] >= 7) or (x[0] == 20 and x[1] <= 6):
    #         ages['20'] = ages['20'] + 1
    #     if (x[0] == 20 and x[1] >= 7) or (x[0] == 21 and x[1] <= 6):
    #         ages['21'] = ages['21'] + 1
    #     if (x[0] == 21 and x[1] >= 7) or (x[0] == 22 and x[1] <= 6):
    #         ages['22'] = ages['22'] + 1
    #
    # print(ages)
    #
    # # Plot età
    # bar_values = [ages['15'], ages['16'], ages['17'], ages['18'], ages['19'], ages['20'], ages['21'], ages['22']]
    # bars = ('15', '16', '17', '18', '19', '20', '21', '22')
    # y_pos = np.arange(len(bars))
    #
    # # Create bars
    # plt.bar(y_pos, bar_values)
    #
    # # Create names on the x-axis
    # plt.xticks(y_pos, bars)
    #
    # plt.title("Dataset di Chieti - Distribuzione dell'Età")
    #
    # # Show graphic
    # plt.show()
    # plt.clf()  # clear plot for next method



    # DATASET MILANO

    ages = {'14' : 0, '15': 0, '16': 0, '17': 0, '18': 0, '19': 0, '20': 0, '21': 0, '22': 0, '23' : 0}

    path = './stadiazione_rx_opt_MI'
    numero_soggetti = 411

    # print(path)
    # print(numero_soggetti)

    sog_list = []

    for index in range(1, (numero_soggetti + 1)):
        # STAMPA

        # print('Soggetto ' + str(index) + ':')

        with open((path + '/soggetto_' + str(index) + '.csv'), newline='') as f:
            reader = csv.reader(f)
            data = list(reader)[0]

        # print(data)

        sog_list.append( [int(data[9]), int(data[10])] )


    print(sog_list)

    # min_year = 100
    # max_year = 0
    #
    # for x in sog_list:
    #     if x[0] < min_year:
    #         min_year = x[0]
    #     if x[0] > max_year:
    #         max_year = x[0]
    #
    # print(min_year)
    # print(max_year)

    for x in sog_list:
        if (x[0] == 13 and x[1] >= 7) or (x[0] == 14 and x[1] <= 6):
            ages['14'] = ages['14'] + 1
        if (x[0] == 14 and x[1] >= 7) or (x[0] == 15 and x[1] <= 6):
            ages['15'] = ages['15'] + 1
        if (x[0] == 15 and x[1] >= 7) or (x[0] == 16 and x[1] <= 6):
            ages['16'] = ages['16'] + 1
        if (x[0] == 16 and x[1] >= 7) or (x[0] == 17 and x[1] <= 6):
            ages['17'] = ages['17'] + 1
        if (x[0] == 17 and x[1] >= 7) or (x[0] == 18 and x[1] <= 6):
            ages['18'] = ages['18'] + 1
        if (x[0] == 18 and x[1] >= 7) or (x[0] == 19 and x[1] <= 6):
            ages['19'] = ages['19'] + 1
        if (x[0] == 19 and x[1] >= 7) or (x[0] == 20 and x[1] <= 6):
            ages['20'] = ages['20'] + 1
        if (x[0] == 20 and x[1] >= 7) or (x[0] == 21 and x[1] <= 6):
            ages['21'] = ages['21'] + 1
        if (x[0] == 21 and x[1] >= 7) or (x[0] == 22 and x[1] <= 6):
            ages['22'] = ages['22'] + 1
        if (x[0] == 22 and x[1] >= 7) or (x[0] == 23 and x[1] <= 6):
            ages['23'] = ages['23'] + 1

    print(ages)

    # Plot età
    bar_values = [ages['14'], ages['15'], ages['16'], ages['17'], ages['18'], ages['19'], ages['20'], ages['21'], ages['22'], ages['23']]
    bars = ('14', '15', '16', '17', '18', '19', '20', '21', '22', '23')
    y_pos = np.arange(len(bars))

    # Create bars
    plt.bar(y_pos, bar_values)

    # Create names on the x-axis
    plt.xticks(y_pos, bars)

    plt.title("Dataset di Milano - Distribuzione dell'Età")

    # Show graphic
    plt.show()
    plt.clf()  # clear plot for next method