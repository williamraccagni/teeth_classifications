import numpy as np
import sklearn.model_selection

from matplotlib import pyplot as plt

def make_dataset(maggiorenni, minorenni):

    X = []
    Y = []

    for index in range(6):

        for j in range(maggiorenni[index]):
            X.append(index)
            Y.append(1)

        for j in range(minorenni[index]):
            X.append(index)
            Y.append(0)

    return X,Y

def predictor(sample, target):

    if sample >= target:
        return 1
    else:
        return 0

if __name__ == '__main__':

    labels_map = ['C', 'D', 'E', 'F', 'G', 'H']

    # # Chieti Maggiore
    # dataset_name = 'Chieti'
    # stage_type = 'Stage Maggiore'
    # maggiorenni = [1,0,3,12,42,77]
    # minorenni = [3,13,11,15,28,13]

    # # Chieti Minore
    # dataset_name = 'Chieti'
    # stage_type = 'Stage Minore'
    # maggiorenni = [1,0,3,13,57,61]
    # minorenni = [7,14,9,22,22,9]

    # # Milano Maggiore
    # dataset_name = 'Milano'
    # stage_type = 'Stage Maggiore'
    # maggiorenni = [0,4,5,19,62,138]
    # minorenni = [24,21,41,34,32,7]

    # Milano Minore
    dataset_name = 'Milano'
    stage_type = 'Stage Minore'
    maggiorenni = [0,6,7,21,74,120]
    minorenni = [30,25,38,35,26,5]

    X, Y = make_dataset(maggiorenni=maggiorenni, minorenni=minorenni)

    for index in range(6):

        predictions = [predictor(x,index) for x in X]

        # METRICS

        print(dataset_name + ' ' + stage_type + ', ' + labels_map[index]+' Confusion Matrix')
        print(sklearn.metrics.confusion_matrix(y_true=Y, y_pred=predictions, labels=np.unique(Y)))

        print(dataset_name + ' ' + stage_type + ', ' + labels_map[index]+' Accuracy:',
              sklearn.metrics.accuracy_score(y_true=Y, y_pred=predictions))

        print(dataset_name + ' ' + stage_type + ', ' + labels_map[index]+' Classification results:')
        print(sklearn.metrics.classification_report(y_true=Y, y_pred=predictions))

        # Confusion Matrix Plot
        matrix = sklearn.metrics.confusion_matrix(y_true=Y, y_pred=predictions, normalize='true')
        print(matrix)

        sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=matrix).plot()  # , display_labels=labels_map).plot()

        plt.title(dataset_name + ' ' + stage_type + ', ' + labels_map[index]+' Confusion Matrix')
        plt.show()
        plt.clf()  # clear plot for next method

