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



