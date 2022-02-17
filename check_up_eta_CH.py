import csv
from datetime import datetime
from matplotlib import pyplot as plt

if __name__ == '__main__':

    # Preparazione Dataset
    path = './stadiazione_rx_opt_CH_v3'
    numero_soggetti = 237

    date_format = "%Y-%m-%d"

    anni_anna = []
    anni_date = []

    for index in range(1, (numero_soggetti + 1) ):
        # Open csv
        with open((path + '/soggetto_' + str(index) + '.csv'), newline='') as f:
            reader = csv.reader(f)
            data = list(reader)[0]

        a = datetime.strptime(data[11], date_format)
        b = datetime.strptime(data[12], date_format)
        delta = b - a
        # print(delta.days)

        print(data)
        print(data[11], data[12], int(delta.days/365.25), int(((delta.days/365.25) - int(delta.days/365.25)) * 12) )

        anna_years = int(data[9]) + (int(data[10])/12)
        anni_anna.append(anna_years)

        date_years = int(delta.days/365.25) + (int(((delta.days/365.25) - int(delta.days/365.25)) * 12)/12)
        anni_date.append(date_years)


    print(anni_anna)
    print(anni_date)

    print('')

    dates_da_confrontare = [ [anni_anna[index], anni_date[index]] for index in range(len(anni_anna)) ]
    print(dates_da_confrontare)
    dates_da_confrontare.sort(key=lambda tup: tup[0])
    print(dates_da_confrontare)

    # visualization

    max_k = len(dates_da_confrontare)
    K = range(1, (max_k + 1))

    # time plot
    plt.plot(K, [x[0] for x in dates_da_confrontare], label="Anni Anna")
    # frequency plot
    plt.plot(K, [x[1] for x in dates_da_confrontare], label="Anni Date")

    plt.legend()

    # plt.xticks(K)
    plt.xlabel('Soggetti')
    plt.ylabel('Anni')
    plt.title('Confronto Anni')
    plt.show()
    # plt.savefig('./k_means/' + filename + '_silhouette_method.png')
    plt.clf()  # clear plot for next method

    # date_format = "%m/%d/%Y"
    # a = datetime.strptime('8/18/2008', date_format)
    # b = datetime.strptime('9/26/2008', date_format)
    # delta = b - a
    # print(delta.days)