from cv2 import cv2, norm
import numpy as np
from matplotlib import pyplot as plt


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


def showImgs(images, scale=1):
    '''
    Given list of images it shows them at given scale
    '''
    counter = 0
    for image in images:
        counter += 1
        print(counter)
        w = int(image.shape[1] / scale)
        h = int(image.shape[0] / scale)
        resized = cv2.resize(image, (w, h))
        cv2.imshow('image {}'.format(counter), resized)
    cv2.waitKey(0)