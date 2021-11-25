from cv2 import cv2, norm
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
from scipy.spatial import distance
import imutils


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    presetWARNING = '\033[93m[WARNING]\033[0m'
    presetINFO = '\033[92m[INFO]\033[0m'
    presetERROR = '\033[91m[ERROR]\033[0m'


def showImgs(images, scale=1):
    '''
    Given list of images it shows them at given scale
    '''
    counter = 0
    for image in images:
        counter += 1
        w = int(image.shape[1] / scale)
        h = int(image.shape[0] / scale)
        resized = cv2.resize(image, (w, h))
        cv2.imshow('image {}'.format(counter), resized)
    cv2.waitKey(0)


def arrayNP2CV(array):
    h, w = array.shape
    array2 = cv2.CreateMat(h, w, cv2.CV_32F)


def downscale(img):

    dim = (200, 200)

    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)


metrics = [(distance.cosine, "min", "COSINE", "PY"), (distance.braycurtis, "min", "BRAYCURTIS", "PY"), (distance.chebyshev, "min", "CHEBYSHEV", "PY", "S"), (distance.minkowski, "min", "MINKOWSKI", "PY"), (distance.euclidean, "min", "EUCLIDIAN", "PY"), (distance.cityblock, "min", "MANATTHAM", "PY"), (cv2.HISTCMP_CORREL, "max", "CORRELATION", "CV"), (cv2.HISTCMP_INTERSECT, "max", "INTERSECTION", "CV"), (cv2.HISTCMP_CHISQR, "min", "CHISQR", "CV"), (cv2.HISTCMP_CHISQR_ALT, "min", "ALTERNATIVE CHISQR", "CV"), (cv2.HISTCMP_BHATTACHARYYA, "min", "BHATTACHARYYA", "CV", "S"), (cv2.HISTCMP_KL_DIV, "min", "KULLBACK-LEIBLER DIVERGENCE", "CV", "S"), (stats.wasserstein_distance, "min", "EARTH MOVER", "PY", "S")]


def maskBoundingBox(mask):
    mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)[1]
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = 0, 0, 0, 0
    for c in contours:
        xc, yc, wc, hc = cv2.boundingRect(c)
        if wc > w and hc > h:
            x, y, w, h = xc, yc, wc, hc

    return x, y, w, h


def cutMask(m):
    x, y, w, h = maskBoundingBox(m)

    m2 = m.copy()
    m3 = m.copy()

    i1 = int(y + w / 3)
    i2 = int(i1 + w / 3)
    m[:, y:i1] = 0
    m2[:, i1:i2] = 0
    m3[:, i2:w] = 0

    return m, m2, m3
