import time
from cv2 import cv2
import numpy as np
from matplotlib import pyplot as plt
from numpy.lib import utils
import segmentation
import yolo
import hist
import videoParsing
import utils
import features
import os
import glob
'''
FULL PIPELINE TEST
'''

im = './files/rider_mask_1.jpg'

histo = hist.computeHist(im, normalize=True)
print(histo.shape)

# out = videoParsing.bboxBackgroundSub('./files/rawVideos/jump11.mp4', 200, 200000, 100, show=False, verbose=True, saveMod=2)
# predictor = segmentation.instantiatePredictor()
# masks = videoParsing.detectronOnVideo('./files/temp/BS_2021-09-30-23-14-52/rider_vide_2.avi', predictor, verbose=True)
# out = videoParsing.bboxBackgroundSub('./files/rawVideos/jump3.mp4', 200, 200000, 100, show=False, verbose=True, saveMod=2)
# out = videoParsing.bboxBackgroundSub('./files/rawVideos/jump4.mp4', 200, 200000, 100, show=False, verbose=True, saveMod=2)
# out = videoParsing.bboxBackgroundSub('./files/rawVideos/jump5.mp4', 200, 200000, 100, show=False, verbose=True, saveMod=2)
# out = videoParsing.bboxBackgroundSub('./files/videos/id1.mp4', 200, 200000, 100, show=False, verbose=True, saveMod=2)
# riders, bs, masks = videoParsing.bboxBackgroundSub('./files/videos/multi1.mp4', 200, 200000, 100, show=False, verbose=True, save=True)

# path = os.path.join(os.getcwd(), "files\\temp\\BS_2021-09-05-17-45-54")

# ridersId = []
# for file in os.listdir(path):
#     rider = cv2.imread(os.path.join(path, file))
#     ridersId.append(rider)

# predictor = segmentation.instantiatePredictor()
# refRider = cv2.imread(".\\files\\temp\\BS_2021-09-05-17-46-52\\rider_2.jpg")
# maskCoco = segmentation.computeSegmentationMask(refRider, predictor, refine=False, CocoClass=0)

# cv2.imshow("MA", maskCoco)
# cv2.waitKey(0)

# histRef = hist.computeHist(refRider, mask=maskCoco, normalize=True)
# cutRef = cv2.bitwise_and(refRider, refRider, mask=maskCoco)

# utils.showImgs(riders, 1)

# print(histRef.shape)

# for x in range(len(ridersId)):
#     # mask = idMasks[x]
#     mask = segmentation.computeSegmentationMask(ridersId[x], predictor, refine=False, CocoClass=0)
#     if not isinstance(mask, bool):
#         cv2.imshow("full", ridersId[x])
#         hist1 = hist.computeHist(ridersId[x], mask=mask, normalize=True)

#         compare = hist.compareHist(histRef, hist1, cv2.HISTCMP_CHISQR)
#         # segm = cv2.bitwise_and(ridersId[x], idBs[x], mask=mask)
#         cutRider = cv2.bitwise_and(ridersId[x], ridersId[x], mask=mask)

#         cv2.imshow("full", ridersId[x])
#         cv2.imshow('cutRef', cutRef)
#         cv2.imshow('ref', refRider)
#         cv2.imshow(f'{compare}', cutRider)

#         cv2.waitKey(0)
'''
FEATURE MATCHING TEST
'''

# img1 = "./files/imgs/green_front.jpg"
# img2 = "./files/imgs/black_front.jpg"

# img1 = cv2.imread(img1)
# img2 = cv2.imread(img2)

# w1 = int(img1.shape[1] / 3)
# h1 = int(img1.shape[0] / 3)
# img1 = cv2.resize(img1, (w1, h1))

# # w2 = int(img2.shape[1] / 3)
# # h2 = int(img2.shape[0] / 3)
# # img2 = cv2.resize(img2, (w2, h2))

# predictor = segmentation.instantiatePredictor()
# mask1 = segmentation.computeSegmentationMask(img1, predictor, refine=False, CocoClass=1)
# mask2 = segmentation.computeSegmentationMask(img2, predictor, refine=False)

# cut = cv2.bitwise_and(img1, img1, mask=mask1)

# utils.showImgs([img1, mask1, img2, mask2, cut], 1)

# # sift = features.istantiateSift()

# # des1, kp1 = features.SiftFeatures(img1, sift, mask=mask1)
# # des2, kp2 = features.SiftFeatures(img2, sift, mask=mask2)

# # matches = features.BFFeatureMatching(des1, des2, cv2.NORM_L2)

# # features.showMatching(img1, img2, kp1, kp2, matches)

# # features.flann(img1, img2, des1, des2, kp1, kp2)
''' TEST '''

# im = ""

# img = cv2.imread(im)
# cv2.imshow(img)
# cv2.waitKey(0)