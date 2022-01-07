import numpy as np
from cv2 import cv2
from caffe.testCaffe import testCaffe
from caffe.Caffe2Pytorch.caffe2pth.caffenet import *


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


def flann(img1, img2, des1, des2, kp1, kp2, min=10):
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if len(good) > min:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
        h, w, d = img1.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
    else:
        print("Not enough matches are found - {}/{}".format(len(good), min))
        matchesMask = None

    draw_params = dict(
        matchColor=(0, 255, 0),  # draw matches in green color
        singlePointColor=None,
        matchesMask=matchesMask,  # draw only inliers
        flags=2)

    img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
    cv2.imshow("FLANN", img3)
    cv2.waitKey(0)


def resNetIstantiateModel():
    model = CaffeNet('caffe/models/ResNet_50/ResNet_50_test.prototxt')
    print(model)
    model.load_state_dict(torch.load('caffe/test.pt'))

    return model