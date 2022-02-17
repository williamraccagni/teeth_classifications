import numpy as np
import cv2
import matplotlib.pyplot as plt
import csv



if __name__ == '__main__':

    # # Otsu
    #
    # img = cv2.imread('./stadiazione_rx_opt_CH/soggetto_1_dx.bmp', 0)
    #
    # # Otsu's thresholding
    # ret, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #
    # plt.subplot(1, 2, 1), plt.imshow(img, cmap='gray')
    # plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(1, 2, 2), plt.imshow(th, cmap='gray')
    # plt.title('Otsu Binary Image'), plt.xticks([]), plt.yticks([])
    # plt.show()




    # KMEANS
    # # original_image = cv2.imread("test.jpeg")
    # original_image = cv2.imread("./stadiazione_rx_opt_CH/soggetto_1_dx.bmp")
    # img = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    #
    # vectorized = img.reshape((-1, 3))
    # vectorized = np.float32(vectorized)
    #
    # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # K = 6
    # attempts = 10
    # ret, label, center = cv2.kmeans(vectorized, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
    #
    # center = np.uint8(center)
    #
    # res = center[label.flatten()]
    # result_image = res.reshape((img.shape))
    #
    # # figure_size = 15
    # # plt.figure(figsize=(figure_size, figure_size))
    # plt.subplot(1, 2, 1), plt.imshow(img)
    # plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(1, 2, 2), plt.imshow(result_image)
    # plt.title('Segmented Image when K = %i' % K), plt.xticks([]), plt.yticks([])
    # plt.show()


    # # CANNY
    # img = cv2.imread("./stadiazione_rx_opt_CH/soggetto_7_dx.bmp",0)
    # edges = cv2.Canny(img, 50, 110)
    # # plt.figure(figsize=(figure_size, figure_size))
    # plt.subplot(1, 2, 1), plt.imshow(img, cmap='gray')
    # plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(1, 2, 2), plt.imshow(edges, cmap='gray')
    # plt.title('Canny Edge Image'), plt.xticks([]), plt.yticks([])
    # plt.show()

    # # DENOISE
    # img = cv2.imread('./stadiazione_rx_opt_CH/soggetto_7_dx.bmp', 0)
    # dst = cv2.fastNlMeansDenoising(src=img)
    # dst = cv2.fastNlMeansDenoising(src=dst)
    # dst = cv2.fastNlMeansDenoising(src=dst)
    # dst = cv2.fastNlMeansDenoising(src=dst)
    # # plt.subplot(121), plt.imshow(img)
    # # plt.subplot(122), plt.imshow(dst)
    # # plt.show()
    # plt.subplot(1, 2, 1), plt.imshow(img, cmap='gray')
    # plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(1, 2, 2), plt.imshow(dst, cmap='gray')
    # plt.title('Denoised Image'), plt.xticks([]), plt.yticks([])
    # plt.show()



    # # EQUALIZATiON
    # img = cv2.imread('./stadiazione_rx_opt_CH/soggetto_7_dx.bmp', 0)
    # equ = cv2.equalizeHist(img)
    # # res = np.hstack((img, equ))  # stacking images side-by-side
    # # cv2.imwrite('res_or_eq.png', res)
    # plt.subplot(1, 2, 1), plt.imshow(img, cmap='gray')
    # plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(1, 2, 2), plt.imshow(equ, cmap='gray')
    # plt.title('Histogram Equalization Image'), plt.xticks([]), plt.yticks([])
    # plt.show()


    # # LOG SOMBRERO
    # img = cv2.imread('./stadiazione_rx_opt_CH/soggetto_7_dx.bmp', 0)
    # # remove noise
    # img = cv2.GaussianBlur(img, (3, 3), 0)
    # # convolute with proper kernels
    # laplacian = cv2.Laplacian(img, cv2.CV_64F)
    # sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)  # x
    # sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)  # y
    # plt.subplot(2, 2, 1), plt.imshow(img, cmap='gray')
    # plt.title('Original'), plt.xticks([]), plt.yticks([])
    # plt.subplot(2, 2, 2), plt.imshow(laplacian, cmap='gray')
    # plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
    # plt.subplot(2, 2, 3), plt.imshow(sobelx, cmap='gray')
    # plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
    # plt.subplot(2, 2, 4), plt.imshow(sobely, cmap='gray')
    # plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
    # plt.show()


    # #SHARPENING
    # img = cv2.imread('./stadiazione_rx_opt_CH/soggetto_7_dx.bmp', 0)
    # # Creating sharpening filter
    # filter = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    # # Applying cv2.filter2D function on our image
    # sharpen_img = cv2.filter2D(img, -1, filter)
    #
    # plt.subplot(1, 2, 1), plt.imshow(img, cmap='gray')
    # plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(1, 2, 2), plt.imshow(sharpen_img, cmap='gray')
    # plt.title('Sharpen Image'), plt.xticks([]), plt.yticks([])
    # plt.show()



    # -----------------------------------------------


    # #FINAL TEST
    #
    # img = cv2.imread('./stadiazione_rx_opt_CH/soggetto_7_dx.bmp', 0)
    # dst = cv2.fastNlMeansDenoising(src=img)
    # dst = cv2.fastNlMeansDenoising(src=dst)
    # dst = cv2.fastNlMeansDenoising(src=dst)
    # dst = cv2.fastNlMeansDenoising(src=dst)
    #
    # # dst = cv2.GaussianBlur(dst, (3, 3), 0)
    #
    # # # Creating sharpening filter
    # # filter = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    # # # Applying cv2.filter2D function on our image
    # # sharpen_img = cv2.filter2D(dst, -1, filter)
    #
    # #canny
    # edges = cv2.Canny(dst, 20, 50)
    #
    # plt.subplot(1, 2, 1), plt.imshow(img, cmap='gray')
    # plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(1, 2, 2), plt.imshow(edges, cmap='gray')
    # plt.title('Test Image'), plt.xticks([]), plt.yticks([])
    # plt.show()




    # --------------------------------------------------------------

    # Bilanciamento dataset

    eta_soggetti = {}

    presenza_denti_dx = {}
    presenza_denti_sx = {}

    stadiazione_dem_dx = {}
    stadiazione_dem_sx = {}

    stadiazione_moo_dx = {}
    stadiazione_moo_sx = {}


    path = './stadiazione_rx_opt_CH'

    for index in range(1,238):

        print('Soggetto '+ str(index) + ':')

        with open((path + '/soggetto_' + str(index) + '.csv'), newline='') as f:
            reader = csv.reader(f)
            data = list(reader)[0]

        print(data)

        if data[1] == 'dente presente' and data[2]=='sconosciuto':
            print('ERRORE - DX')
        if data[4] == 'dente presente' and data[5]=='sconosciuto':
            print('ERRORE - SX')

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



        # stadiazione_dem_dx
        if data[2] not in stadiazione_dem_dx.keys():
            stadiazione_dem_dx[data[2]] = 1
        else:
            stadiazione_dem_dx[data[2]] = stadiazione_dem_dx[data[2]] + 1

        # stadiazione_dem_sx
        if data[5] not in stadiazione_dem_sx.keys():
            stadiazione_dem_sx[data[5]] = 1
        else:
            stadiazione_dem_sx[data[5]] = stadiazione_dem_sx[data[5]] + 1



        # stadiazione_moo_dx
        if data[3] not in stadiazione_moo_dx.keys():
            stadiazione_moo_dx[data[3]] = 1
        else:
            stadiazione_moo_dx[data[3]] = stadiazione_moo_dx[data[3]] + 1

        # stadiazione_moo_sx
        if data[6] not in stadiazione_moo_sx.keys():
            stadiazione_moo_sx[data[6]] = 1
        else:
            stadiazione_moo_sx[data[6]] = stadiazione_moo_sx[data[6]] + 1


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