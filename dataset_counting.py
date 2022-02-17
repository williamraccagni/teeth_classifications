import csv
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    percorsi_doppi = []
    percorsi = []
    errori = []

    # Bilanciamento dataset
    eta_soggetti = {}

    presenza_denti_dx = {}
    presenza_denti_sx = {}

    stadiazione_dem_dx = {}
    stadiazione_dem_sx = {}

    stadiazione_moo_dx = {}
    stadiazione_moo_sx = {}


    path = './stadiazione_rx_opt_CH'
    numero_soggetti = 237

    print(path)
    print(numero_soggetti)

    for index in range(1, (numero_soggetti + 1) ):

        # STAMPA

        print('Soggetto '+ str(index) + ':')

        with open((path + '/soggetto_' + str(index) + '.csv'), newline='') as f:
            reader = csv.reader(f)
            data = list(reader)[0]

        print(data)



        # CHECK ERRORI

        if data[1] == 'dente non presente' and (data[2]!='sconosciuto' or data[3]!='sconosciuto'):
            print('ERRORE - DX')
            errori.append(index)
        if data[4] == 'dente non presente' and (data[5]!='sconosciuto' or data[6]!='sconosciuto'):
            print('ERRORE - SX')
            errori.append(index)



        if data[1] == 'dente presente' and ( (data[2]!='sconosciuto' and data[3]=='sconosciuto') or
                                             (data[2]=='sconosciuto' and data[3]!='sconosciuto') ):
            print('ERRORE 2 - DX')
            errori.append(index)
        if data[4] == 'dente presente' and ( (data[5]!='sconosciuto' and data[6]=='sconosciuto') or
                                             (data[5]=='sconosciuto' and data[6]!='sconosciuto') ):
            print('ERRORE 2 - SX')
            errori.append(index)



        if data[13] not in percorsi:
            percorsi.append(data[13])
        else:
            percorsi_doppi.append(data[13])



        #CONTEGGIO

        # eta_soggetti
        if data[9] not in eta_soggetti.keys():
            eta_soggetti[data[9]] = 1
        else:
            eta_soggetti[data[9]] = eta_soggetti[data[9]] + 1



        # presenza_denti_dx
        if data[1] not in presenza_denti_dx.keys():
            presenza_denti_dx[data[1]] = 1
        else:
            presenza_denti_dx[data[1]] = presenza_denti_dx[data[1]] + 1

        # presenza_denti_sx
        if data[4] not in presenza_denti_sx.keys():
            presenza_denti_sx[data[4]] = 1
        else:
            presenza_denti_sx[data[4]] = presenza_denti_sx[data[4]] + 1



        if data[1] == 'dente presente':
            # stadiazione_dem_dx
            if data[2] not in stadiazione_dem_dx.keys():
                stadiazione_dem_dx[data[2]] = 1
            else:
                stadiazione_dem_dx[data[2]] = stadiazione_dem_dx[data[2]] + 1


        if data[4] == 'dente presente':
            # stadiazione_dem_sx
            if data[5] not in stadiazione_dem_sx.keys():
                stadiazione_dem_sx[data[5]] = 1
            else:
                stadiazione_dem_sx[data[5]] = stadiazione_dem_sx[data[5]] + 1





        if data[1] == 'dente presente':
            # stadiazione_moo_dx
            if data[3] not in stadiazione_moo_dx.keys():
                stadiazione_moo_dx[data[3]] = 1
            else:
                stadiazione_moo_dx[data[3]] = stadiazione_moo_dx[data[3]] + 1

        if data[4] == 'dente presente':
            # stadiazione_moo_sx
            if data[6] not in stadiazione_moo_sx.keys():
                stadiazione_moo_sx[data[6]] = 1
            else:
                stadiazione_moo_sx[data[6]] = stadiazione_moo_sx[data[6]] + 1





    # PRINT INFO DATASET
    print(path)
    print(numero_soggetti)

    # print errori
    print(errori)
    print(percorsi_doppi)



    print('Conteggio Etichette:')
    print('---Età---')
    for x in eta_soggetti.keys():
        print(x + ': ' + str(eta_soggetti[x]) )

    print('---PresenzaDentiDx---')
    for x in presenza_denti_dx.keys():
        print(x + ': ' + str(presenza_denti_dx[x]) )
    print('---PresenzaDentiSx---')
    for x in presenza_denti_sx.keys():
        print(x + ': ' + str(presenza_denti_sx[x]) )

    print('---stadiazione_dem_dx---')
    for x in stadiazione_dem_dx.keys():
        print(x + ': ' + str(stadiazione_dem_dx[x]) )
    print('---stadiazione_dem_sx---')
    for x in stadiazione_dem_sx.keys():
        print(x + ': ' + str(stadiazione_dem_sx[x]) )

    print('---stadiazione_moo_dx---')
    for x in stadiazione_moo_dx.keys():
        print(x + ': ' + str(stadiazione_moo_dx[x]) )
    print('---stadiazione_moo_sx---')
    for x in stadiazione_moo_sx.keys():
        print(x + ': ' + str(stadiazione_moo_sx[x]) )


# -----------------------------------------------------------------------------------

    # PLOT DISTRIBUZIONI Chieti

    # city_name = 'Chieti'



    # Plot età
    # bar_values = [31, 36, 38, 29, 36, 40, 27]
    # bars = ('16', '17', '18', '19', '20', '21', '22')
    # y_pos = np.arange(len(bars))
    #
    # # Create bars
    # plt.bar(y_pos, bar_values)
    #
    # # Create names on the x-axis
    # plt.xticks(y_pos, bars)
    #
    # plt.title('Dataset di '+city_name+" - Distribuzione dell'Età")
    #
    # # Show graphic
    # plt.show()
    # plt.clf()  # clear plot for next method



    # # Distribuzione Denti DX
    #
    # bar_values = [206, 31]
    # bars = ('DP', 'DNP')
    # y_pos = np.arange(len(bars))
    #
    # # Create bars
    # plt.bar(y_pos, bar_values)
    #
    # # Create names on the x-axis
    # plt.xticks(y_pos, bars)
    #
    # plt.title('Dataset di '+city_name+" - Distribuzione del Terzo Molare Destro")
    #
    # # Show graphic
    # plt.show()
    # plt.clf()  # clear plot for next method
    #
    #
    #
    # # Distribuzione Denti SX
    #
    # bar_values = [207, 30]
    # bars = ('DP', 'DNP')
    # y_pos = np.arange(len(bars))
    #
    # # Create bars
    # plt.bar(y_pos, bar_values)
    #
    # # Create names on the x-axis
    # plt.xticks(y_pos, bars)
    #
    # plt.title('Dataset di ' + city_name + " - Distribuzione del Terzo Molare Sinistro")
    #
    # # Show graphic
    # plt.show()
    # plt.clf()  # clear plot for next method




    # # Distribuzione Stadi Dem DX
    #
    # bar_values = [0, 0, 3, 17, 9, 26, 78, 71, 2]
    # bars = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'Sconosciuto')
    # y_pos = np.arange(len(bars))
    #
    # # Create bars
    # plt.bar(y_pos, bar_values)
    #
    # # Create names on the x-axis
    # plt.xticks(y_pos, bars)
    #
    # plt.title('Dataset di ' + city_name + " - Distribuzione degli Stage (met. Demirjian)\ndel Terzo Molare Destro")
    #
    # # Show graphic
    # plt.show()
    # plt.clf()  # clear plot for next method
    #
    #
    #
    # # Distribuzione Stadi Dem SX
    #
    # bar_values = [0, 0, 7, 8, 15, 29, 65, 80, 3]
    # bars = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'Sconosciuto')
    # y_pos = np.arange(len(bars))
    #
    # # Create bars
    # plt.bar(y_pos, bar_values)
    #
    # # Create names on the x-axis
    # plt.xticks(y_pos, bars)
    #
    # plt.title('Dataset di ' + city_name + " - Distribuzione degli Stage (met. Demirjian)\ndel Terzo Molare Sinistro")
    #
    # # Show graphic
    # plt.show()
    # plt.clf()  # clear plot for next method



    # # Distribuzione Stadi Moo DX
    #
    # bar_values = [0,0,0,0,0, 3, 17, 9, 11, 16, 35, 42, 71, 2]
    # bars = ('Ci', 'Cco', 'Coc', 'Cr 1/2', 'Cr 3/4', 'Crc', 'Ri', 'R 1/4', 'R 1/2', 'R 3/4', 'Rc', 'A 1/2', 'Ac', 'Scon.')
    # y_pos = np.arange(len(bars))
    #
    # plt.figure(figsize=(8, 4))
    #
    # # Create bars
    # plt.bar(y_pos, bar_values)
    #
    # # Create names on the x-axis
    # plt.xticks(y_pos, bars)
    #
    # plt.title('Dataset di ' + city_name + " - Distribuzione degli Stage (met. Moorrees)\ndel Terzo Molare Destro")
    #
    # # Show graphic
    # plt.show()
    # plt.clf()  # clear plot for next method
    #
    #
    #
    # # Distribuzione Stadi Moo SX
    #
    # bar_values = [0,0,0,0, 1, 6, 8, 15, 13, 16, 31, 34, 80, 3]
    # bars = ('Ci', 'Cco', 'Coc', 'Cr 1/2', 'Cr 3/4', 'Crc', 'Ri', 'R 1/4', 'R 1/2', 'R 3/4', 'Rc', 'A 1/2', 'Ac', 'Scon.')
    # y_pos = np.arange(len(bars))
    #
    # plt.figure(figsize=(8, 4))
    #
    # # Create bars
    # plt.bar(y_pos, bar_values)
    #
    # # Create names on the x-axis
    # plt.xticks(y_pos, bars)
    #
    # plt.title('Dataset di ' + city_name + " - Distribuzione degli Stage (met. Moorrees)\ndel Terzo Molare Sinistro")
    #
    # # Show graphic
    # plt.show()
    # plt.clf()  # clear plot for next method


# -----------------------------------------------------------------------------------------------------------------

    # PLOT DISTRIBUZIONI Milano

    city_name = 'Milano'

    # # Plot età
    # bar_values = [29, 49, 48, 47, 40, 53, 44, 52, 49]
    # bars = ('14', '15', '16', '17', '18', '19', '20', '21', '22')
    # y_pos = np.arange(len(bars))
    #
    # # Create bars
    # plt.bar(y_pos, bar_values)
    #
    # # Create names on the x-axis
    # plt.xticks(y_pos, bars)
    #
    # plt.title('Dataset di '+city_name+" - Distribuzione dell'Età")
    #
    # # Show graphic
    # plt.show()
    # plt.clf()  # clear plot for next method
    #
    # # Distribuzione Denti DX
    #
    # bar_values = [378, 33]
    # bars = ('DP', 'DNP')
    # y_pos = np.arange(len(bars))
    #
    # # Create bars
    # plt.bar(y_pos, bar_values)
    #
    # # Create names on the x-axis
    # plt.xticks(y_pos, bars)
    #
    # plt.title('Dataset di '+city_name+" - Distribuzione del Terzo Molare Destro")
    #
    # # Show graphic
    # plt.show()
    # plt.clf()  # clear plot for next method
    #
    #
    #
    # # Distribuzione Denti SX
    #
    # bar_values = [381, 30]
    # bars = ('DP', 'DNP')
    # y_pos = np.arange(len(bars))
    #
    # # Create bars
    # plt.bar(y_pos, bar_values)
    #
    # # Create names on the x-axis
    # plt.xticks(y_pos, bars)
    #
    # plt.title('Dataset di ' + city_name + " - Distribuzione del Terzo Molare Sinistro")
    #
    # # Show graphic
    # plt.show()
    # plt.clf()  # clear plot for next method



    # # Distribuzione Stadi Dem DX
    #
    # bar_values = [0, 0, 22, 25, 47, 50, 97, 123, 14]
    # bars = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'Sconosciuto')
    # y_pos = np.arange(len(bars))
    #
    # # Create bars
    # plt.bar(y_pos, bar_values)
    #
    # # Create names on the x-axis
    # plt.xticks(y_pos, bars)
    #
    # plt.title('Dataset di ' + city_name + " - Distribuzione degli Stage (met. Demirjian)\ndel Terzo Molare Destro")
    #
    # # Show graphic
    # plt.show()
    # plt.clf()  # clear plot for next method
    #
    #
    #
    # # Distribuzione Stadi Dem SX
    #
    # bar_values = [0,0, 26, 25, 41, 54, 88, 132, 15]
    # bars = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'Sconosciuto')
    # y_pos = np.arange(len(bars))
    #
    # # Create bars
    # plt.bar(y_pos, bar_values)
    #
    # # Create names on the x-axis
    # plt.xticks(y_pos, bars)
    #
    # plt.title('Dataset di ' + city_name + " - Distribuzione degli Stage (met. Demirjian)\ndel Terzo Molare Sinistro")
    #
    # # Show graphic
    # plt.show()
    # plt.clf()  # clear plot for next method



    # # Distribuzione Stadi Moo DX
    #
    # bar_values = [0,0,0,0, 6, 16, 25, 47, 17, 37, 29, 64, 123, 14]
    # bars = ('Ci', 'Cco', 'Coc', 'Cr 1/2', 'Cr 3/4', 'Crc', 'Ri', 'R 1/4', 'R 1/2', 'R 3/4', 'Rc', 'A 1/2', 'Ac', 'Scon.')
    # y_pos = np.arange(len(bars))
    #
    # plt.figure(figsize=(8, 4))
    #
    # # Create bars
    # plt.bar(y_pos, bar_values)
    #
    # # Create names on the x-axis
    # plt.xticks(y_pos, bars)
    #
    # plt.title('Dataset di ' + city_name + " - Distribuzione degli Stage (met. Moorrees)\ndel Terzo Molare Destro")
    #
    # # Show graphic
    # plt.show()
    # plt.clf()  # clear plot for next method
    #
    #
    #
    # # Distribuzione Stadi Moo SX
    #
    # bar_values = [0,0,0,0, 9, 16, 25, 41, 22, 34, 29, 58, 132, 15]
    # bars = ('Ci', 'Cco', 'Coc', 'Cr 1/2', 'Cr 3/4', 'Crc', 'Ri', 'R 1/4', 'R 1/2', 'R 3/4', 'Rc', 'A 1/2', 'Ac', 'Scon.')
    # y_pos = np.arange(len(bars))
    #
    # plt.figure(figsize=(8, 4))
    #
    # # Create bars
    # plt.bar(y_pos, bar_values)
    #
    # # Create names on the x-axis
    # plt.xticks(y_pos, bars)
    #
    # plt.title('Dataset di ' + city_name + " - Distribuzione degli Stage (met. Moorrees)\ndel Terzo Molare Sinistro")
    #
    # # Show graphic
    # plt.show()
    # plt.clf()  # clear plot for next method