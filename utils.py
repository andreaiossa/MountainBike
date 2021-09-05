from cv2 import cv2, norm
import numpy as np
from matplotlib import pyplot as plt


def showImgs(images, scale):
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