import cv2
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from scipy.spatial import distance
import sys
from components.utils import *
import seaborn as sns
from tabulate import tabulate

matplotlib.use('TkAgg')

def compareHistCV(hist1, hist2, metric):
    '''
    Gives the result of the comparison of two histograms with defined metric.
    METRICS: CV_COMP_CORREL (best is higher), CV_COMP_INTERSECT (best is higher), CV_COMP_CHISQR (best is lower), CV_COMP_BHATTACHARYYA (best is lower)
    '''

    return cv2.compareHist(hist1.flatten(), hist2.flatten(), metric)


def compareHistPY(hist1, hist2, metric):
    '''
    Gives the result of the comparison of two histograms with defined metric.
    METRICS: CV_COMP_CORREL (best is higher), CV_COMP_INTERSECT (best is higher), CV_COMP_CHISQR (best is lower), CV_COMP_BHATTACHARYYA (best is lower)
    '''

    return metric(hist1.flatten(), hist2.flatten())


def compute2DHist(img, mask=None, normalize=False, difference=False):
    '''
    Given img (BGR) and mask return the 8 bins histogram in HS color space
    '''
    if isinstance(mask, bool):
        print(f"{bcolors.presetWARNING} Impossible to compute histograms, given mask is empty")
        return

    img = cv2.imread(img) if isinstance(img, str) else img
    static_image_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    channels = [0, 1]
    h_ranges = [0, 180]
    s_ranges = [0, 256]
    ranges = h_ranges + s_ranges

    hist = cv2.calcHist([static_image_HSV], channels, mask, [64, 32], ranges, accumulate=False)
    if not isinstance(difference, bool):
        hist = diffHist(hist, difference)

    if normalize == "density":
        if hist.sum() > 0:
            hist = hist / hist.sum()

    elif normalize == cv2.NORM_MINMAX:
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=normalize)

    elif normalize:
        cv2.normalize(hist, hist, norm_type=normalize)

    return hist


def compute1DHist(img, mask=None, normalize=False):
    '''
    Given img (BGR) and mask return the 8 bins histogram for H and for S of HSV (separatly)
    '''
    if isinstance(mask, bool):
        print(f"{bcolors.presetWARNING} Impossible to compute histograms, given mask is empty")
        return

    img = cv2.imread(img) if isinstance(img, str) else img
    static_image_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    h_ranges = [0, 180]
    s_ranges = [0, 256]

    histH = cv2.calcHist([static_image_HSV], [0], mask, [8], h_ranges, accumulate=False)
    histS = cv2.calcHist([static_image_HSV], [1], mask, [8], s_ranges, accumulate=False)

    if normalize == "density":
        if histH.sum() > 0:
            histH = histH / histH.sum()

    elif normalize == cv2.NORM_MINMAX:
        cv2.normalize(histH, histH, alpha=0, beta=1, norm_type=normalize)
        cv2.normalize(histS, histS, alpha=0, beta=1, norm_type=normalize)

    elif normalize:
        cv2.normalize(histH, histH, norm_type=normalize)
        cv2.normalize(histS, histS, norm_type=normalize)

    return histH, histS


def softMaxHist(hist):
    histExp = np.exp(hist)

    return histExp / histExp.sum()


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
            print(f"{bcolors.presetERROR} Histograms need to be of same shape (bins) to average them")
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
        print(f"{bcolors.presetERROR} Histograms need to be of same shape (bins) to difference them")
        return
    shape = hist1.shape
    outHist = np.zeros(shape)
    for x in range(shape[0]):
        for y in range(shape[1]):
            a = hist1[x][y]
            b = hist2[x][y]
            outHist[x][y] = a - b if a - b > 0 else 0

    return outHist

def histNormalize(hist, normalize=cv2.NORM_MINMAX):
    if normalize == "density":
        hist = hist / hist.sum()

    elif normalize == cv2.NORM_MINMAX:
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=normalize)

    elif normalize:
        cv2.normalize(hist, hist, norm_type=normalize)

    return hist