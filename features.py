import numpy as np
from cv2 import cv2


def istantiateSift():
    '''
    Instantiate a SIFT Feature Extractor
    '''
    print("[INFO] Creating Sift Feature Extractor...")
    sift = cv2.SIFT_create()
    print("[INFO] Done")

    return sift


def SiftFeatures(img, extractor, mask=None):
    '''
    Given an imge (or path to image) and an extractor (DIFT, SURF, ORB) extract it returns the descriptor of the image. Possible to pass a mask to analyze only ROI of the image
    '''
    img = cv2.imread(img) if isinstance(img, str) else img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    keyPoints, descriptor = extractor.detectAndCompute(gray, mask)

    return descriptor, keyPoints


def BFFeatureMatching(des1, des2, normalization):
    '''
    Given two descriptors it applies a Brute Force matching. Use cv2.NORM_L2 for SIFT and SURF, cv2.NORM_HAMMING for ORB BRIEF etc. Return a a list of DMatch objects. This DMatch object has following attributes:
    DMatch.distance - Distance between descriptors. The lower, the better it is.
    DMatch.trainIdx - Index of the descriptor in train descriptors
    DMatch.queryIdx - Index of the descriptor in query descriptors
    DMatch.imgIdx - Index of the train image.   
    '''

    bf = cv2.BFMatcher(normalization, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    return matches


def showMatching(img1, img2, kp1, kp2, matches):

    img1 = cv2.imread(img1) if isinstance(img1, str) else img1
    img2 = cv2.imread(img2) if isinstance(img2, str) else img2

    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow("Matching", img3)

    cv2.waitKey(0)
