import time
import cv2
import numpy as np
from matplotlib import pyplot as plt
import preprocess
import pickle
import sys
from tabulate import tabulate
import hist

riders = preprocess.collectRiders()
riders = sorted(riders, key=lambda x: int(x.name.split("RIDER")[1]))
'''
FULL PIPELINE TEST
'''

for rider in riders:
    rider.collectMasks(mod="bSub")
    rider.collectMasks(video=rider.customVid, mod="bSub")
    # rider.collectHists(mod="identification")
    # rider.collectHists(mod="custom")
    # rider.squashHist(mod="median")
    # preprocess.updateRider(rider)

# zero = pickle.load(open("./files/pickles/RIDER2.p", "rb"))

# print(len(riders[1].frameAndMasksCustom))
preprocess.checkMasks(riders[1].frameAndMasksCustom)
# print(len(riders[1].frameAndMasksBack))
preprocess.checkMasks(riders[1].frameAndMasksBack)

# hist.fullHistComp(riders, "bSub_64_H.txt")
# hist.fullHistComp(riders, "bSub_64_HS.txt", channels=2)
''' SHOW BACK HISTOGRAMS'''

# for rider in riders:
#     fig1 = plt.figure(f"{rider.name}")
#     hist.displayHist(rider.backHist2D, fig1, mod=1)
#     plt.show()
''' SHOW FRAME RIDER'''
# for rider in riders:
#     if rider.name == "RIDER1":
#         preprocess.checkDetectron(rider.frameAndMasksCustom)

# for rider in riders:
#     if rider.name == "RIDER1":
#         preprocess.checkDetectron(rider.frameAndMasksCustom)

# mask = "files/temp/testMask.jpg"
# img = "files/temp/testImg.jpg"

# img = cv2.imread(img)
# mask = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
# outputMask = np.where((mask > 1), 1, 0)
# outputMask = (outputMask * 255).astype("uint8")
# kernel = np.ones((5, 5), np.uint8)

# closing1 = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
# closing2 = cv2.morphologyEx(outputMask, cv2.MORPH_CLOSE, kernel)

# cut = cv2.bitwise_and(img, img, mask=mask)
# cut1 = cv2.bitwise_and(img, img, mask=outputMask)
# cut2 = cv2.bitwise_and(img, img, mask=closing1)
# cut3 = cv2.bitwise_and(img, img, mask=closing2)

# cv2.imshow("fg", outputMask)
# cv2.imshow("closing1", closing1)
# cv2.imshow("closing2", closing2)
# cv2.imshow("cut fg", cut1)
# cv2.imshow("cut closing1", cut2)
# cv2.imshow("cut closing2", cut3)
# cv2.imshow("original", cut)
# cv2.waitKey(0)
