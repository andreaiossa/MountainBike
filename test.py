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
'''
FULL PIPELINE TEST
'''

# ridersId, idBs, idMasks = videoParsing.bboxBackgroundSub('./files/videos/id1.mp4', 200, 200000, 100, show=False, verbose=True)
# riders, bs, masks = videoParsing.bboxBackgroundSub('./files/videos/multi1.mp4', 200, 200000, 100, show=False, verbose=True)

# # predictor = segmentation.instantiatePredictor()
# refRider = riders[1]
# # maskTest = segmentation.computeSegmentationMask(refRider, predictor, refine=True)
# histRef = hist.computeHist(riders[1], mask=masks[1], normalize=True)

# # utils.showImgs(riders, 1)

# for x in range(len(ridersId)):
#     mask = idMasks[x]
#     # mask = segmentation.computeSegmentationMask(rider, predictor, refine=False)
#     if not isinstance(mask, bool):
#         hist1 = hist.computeHist(idBs[x], mask=mask, normalize=True)

#         compare = hist.compareHist(histRef, hist1, cv2.HISTCMP_CHISQR)
#         # segm = cv2.bitwise_and(ridersId[x], idBs[x], mask=mask)

#         cv2.imshow("full", ridersId[x])
#         cv2.imshow(f'ref', refRider)
#         cv2.imshow(f'ref2', bs[1])
#         cv2.imshow(f'{compare}', idBs[x])

#         cv2.waitKey(0)
'''
FEATURE MATCHING TEST
'''

img1 = "./files/imgs/green_back.jpg"
img2 = "./files/imgs/green_back.jpg"

img1 = cv2.imread(img1)
img2 = cv2.imread(img2)

w1 = int(img1.shape[1] / 3)
h1 = int(img1.shape[0] / 3)
img1 = cv2.resize(img1, (w1, h1))

w2 = int(img2.shape[1] / 3)
h2 = int(img2.shape[0] / 3)
img2 = cv2.resize(img2, (w2, h2))

predictor = segmentation.instantiatePredictor()
mask1 = segmentation.computeSegmentationMask(img1, predictor, refine=False)
mask2 = segmentation.computeSegmentationMask(img2, predictor, refine=True)

utils.showImgs([img1, mask1, img2, mask2], 1)

sift = features.istantiateSift()

des1, kp1 = features.SiftFeatures(img1, sift, mask=mask1)
des2, kp2 = features.SiftFeatures(img2, sift, mask=mask2)

matches = features.BFFeatureMatching(des1, des2, cv2.NORM_L2)

features.showMatching(img1, img2, kp1, kp2, matches)