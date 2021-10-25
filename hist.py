import cv2
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from scipy.spatial import distance
import sys
import utils
import seaborn as sns
from tabulate import tabulate

matplotlib.use('TkAgg')


def compareHistCV(hist1, hist2, metric):
    '''
    Gives the result of the comparison of two histograms with defined metric.
    METRICS: CV_COMP_CORREL (best is higher), CV_COMP_INTERSECT (best is higher), CV_COMP_CHISQR (best is lower), CV_COMP_BHATTACHARYYA (best is lower)
    '''
    result = cv2.compareHist(hist1.flatten(), hist2.flatten(), metric)
    return result


def compareHistPY(hist1, hist2, metric):
    '''
    Gives the result of the comparison of two histograms with defined metric.
    METRICS: CV_COMP_CORREL (best is higher), CV_COMP_INTERSECT (best is higher), CV_COMP_CHISQR (best is lower), CV_COMP_BHATTACHARYYA (best is lower)
    '''
    result = metric(hist1.flatten(), hist2.flatten())
    return result


def compute2DHist(img, mask=None, normalize=False, difference=False, pixels=None):
    '''
    Given img (BGR) and mask return the 8 bins histogram in HS color space
    '''
    if isinstance(mask, bool):
        print(f"{utils.bcolors.presetWARNING} Impossible to compute histograms, given mask is empty")
        return

    img = cv2.imread(img) if isinstance(img, str) else img
    static_image_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    channels = [0, 1]
    h_ranges = [0, 180]
    s_ranges = [0, 256]
    ranges = h_ranges + s_ranges

    hist = cv2.calcHist([static_image_HSV], channels, mask, [8, 8], ranges, accumulate=False)
    if not isinstance(difference, bool):
        # print("\nHIST BEFORE \n", hist)
        # print("\n WOODS \n", hist)
        hist = diffHist(hist, difference)
        # print("\nHIST after \n", hist)

    if normalize:
        # cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(hist, hist, norm_type=cv2.NORM_L2)
        # denominator = pixels if pixels else hist.sum()
        #hist = hist / hist.sum()

    return hist


def compute1DHist(img, mask=None, normalize=False):
    '''
    Given img (BGR) and mask return the 8 bins histogram for H and for S of HSV (separatly)
    '''
    if isinstance(mask, bool):
        print(f"{utils.bcolors.presetWARNING} Impossible to compute histograms, given mask is empty")
        return

    img = cv2.imread(img) if isinstance(img, str) else img
    static_image_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    h_ranges = [0, 180]
    s_ranges = [0, 256]

    histH = cv2.calcHist([static_image_HSV], [0], mask, [8], h_ranges, accumulate=False)
    histS = cv2.calcHist([static_image_HSV], [1], mask, [8], s_ranges, accumulate=False)

    if normalize:
        cv2.normalize(histH, histH, norm_type=cv2.NORM_L2)
        cv2.normalize(histS, histS, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    return (histH, histS)


def displayHist(hist, fig, mod=1):
    # fig = plt.figure()
    ax1 = fig.add_subplot()
    if mod == 0:
        colors = ("b", "g", "r")
        for h, c in zip(hist, colors):
            ax1.plot(h, color=c)

    if mod == 1:
        ax1 = sns.heatmap(hist, vmin=0, vmax=1)

    if mod == 2:
        data_array = np.array(hist)
        x_data, y_data = np.meshgrid(np.arange(data_array.shape[1]), np.arange(data_array.shape[0]))
        x_data = x_data.flatten()
        y_data = y_data.flatten()
        z_data = data_array.flatten()

        ax2 = fig.add_subplot(111, projection='3d')
        ax2.bar3d(x_data, y_data, np.zeros(len(z_data)), 1, 1, z_data)


def squashHists(hists, mod="median"):
    shape = hists[0].shape
    for hist in hists:
        if hist.shape != shape:
            print(f"{utils.bcolors.presetERROR} Histograms need to be of same shape (bins) to average them")
            return
    outHist = np.zeros(shape)
    if mod == "median":
        for x in range(shape[0]):
            for y in range(shape[1]):
                values = []
                for hist in hists:
                    values.append(hist[x][y])
                finalValue = np.median(values)
                outHist[x][y] = finalValue

    if mod == "mean":
        for x in range(shape[0]):
            for y in range(shape[1]):
                values = []
                for hist in hists:
                    values.append(hist[x][y])
                finalValue = np.mean(values)
                outHist[x][y] = finalValue

    return outHist


def diffHist(hist1, hist2):

    if hist1.shape != hist2.shape:
        print(f"{utils.bcolors.presetERROR} Histograms need to be of same shape (bins) to difference them")
        return
    shape = hist1.shape
    outHist = np.zeros(shape)
    for x in range(shape[0]):
        for y in range(shape[1]):
            a = hist1[x][y]
            b = hist2[x][y]
            outHist[x][y] = a - b if a - b > 0 else 0

    return outHist


def fullHistComp(riders, fileName, channels=1, show=False):
    tables = []
    headers = []
    for rider in riders:
        headers.append(rider.name)

    for metricTuple in utils.metrics:
        metric = metricTuple[0]
        mod = metricTuple[1]
        metricName = metricTuple[2]
        fun = metricTuple[3]
        score = 0
        table = []
        for ref in riders:
            tmpRow = []
            row = [ref.name]
            minimum = None
            maximum = None
            match = False
            shouldMatch = None
            for rider in riders:
                if channels == 1:
                    refHist = np.float32(ref.backHist1D)
                    riderHist = np.float32(rider.customHist1D)
                if channels == 2:
                    refHist = np.float32(ref.backHist2D)
                    riderHist = np.float32(rider.customHist2D)
                if show:
                    fig1 = plt.figure(f"REF: {ref.name}")
                    displayHist(refHist, fig1, mod=1)
                    plt.show()
                    fig1 = plt.figure(rider.name)
                    displayHist(riderHist, fig1, mod=1)
                    plt.show()
                if fun == "CV":
                    result = compareHistCV(riderHist, refHist, metric)
                elif fun == "PY":
                    result = compareHistPY(riderHist, refHist, metric)
                if mod == "min":
                    if not minimum:
                        minimum = result
                        match = rider
                    elif result < minimum:
                        minimum = result
                        match = rider
                if mod == "max":
                    if not maximum:
                        maximum = result
                        match = rider
                    elif result > maximum:
                        maximum = result
                        match = rider
                if rider.name == ref.name:
                    shouldMatch = result

                tmpRow.append(result)

            best = minimum if mod == "min" else maximum
            score = score + 1 if ref.name == match.name else score
            for r in tmpRow:
                if r == shouldMatch and r == best:
                    row.append(f"{utils.bcolors.OKBLUE}{r}{utils.bcolors.ENDC}")
                elif r == best:
                    row.append(f"{utils.bcolors.OKCYAN}{r}{utils.bcolors.ENDC}")
                else:
                    row.append(r)
            table.append(row)

        table.append([f"{utils.bcolors.OKGREEN}{metricName}", f"Score: {score}/10", f"better is {mod}{utils.bcolors.ENDC}"])
        tables.append(table)

    original_stdout = sys.stdout
    with open(fileName, 'w') as f:
        sys.stdout = f
        for tab in tables:
            print(tabulate(tab, headers=headers))
            print("\n")
        sys.stdout = original_stdout