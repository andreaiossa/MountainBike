from cv2 import cv2, norm
import numpy as np
from matplotlib import pyplot as plt
import segmentation
import yolo


def compareHist(hist1, hist2, metric):
    '''
    Gives the result of the comparison of two histograms with defined metric.
    METRICS: CV_COMP_CORREL (best is higher), CV_COMP_INTERSECT (best is higher), CV_COMP_CHISQR (best is lower), CV_COMP_BHATTACHARYYA (best is lower)
    '''
    result = cv2.compareHist(hist1, hist2, metric)
    return result


def computeHist(img, mask=None, normalize=False):
    '''
    Given img (BGR) and mask return the 8 bins histogram in HS color space
    '''
    img = cv2.imread(img) if isinstance(img, str) else img
    static_image_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    channels = [0, 1]
    h_ranges = [0, 180]
    s_ranges = [0, 256]
    ranges = h_ranges + s_ranges

    hist = cv2.calcHist([static_image_HSV], channels, mask, [8, 8], ranges, accumulate=False)

    if normalize:
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    return hist
