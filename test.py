import time
import cv2
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import preprocess
import pickle
import sys
from tabulate import tabulate
import hist
import videoParsing
from scipy.spatial import distance
import utils
import segmentation

### CREATE EMPYT RIDERS FROM FOLDER

# RIDERS = preprocess.collectRidersFolders()
# preprocess.processRider(RIDERS)

### IMPORT RIDERS PICKLES AND PROCESS

riders = preprocess.collectRiders()
riders = sorted(riders, key=lambda x: int(x.name.split("RIDER")[1]))

# for rider in riders:

#     # rider.processFiles(customFile="jump")
#     # rider.collectMasksCancelletto()
#     # rider.collectMasksVid()
#     rider.collectHistsCancelletto(mod="standard", normalization=cv2.NORM_L2)
#     rider.collectHistsVid(mod="standard", normalization=cv2.NORM_L2)
#     rider.squashHist(mod="median")
#     preprocess.updateRider(rider)

# --------------------------------------------------------------------------------------- COMPARISON ------------------------------------------------------------------------------------------------------------------------------

# hist.fullHistComp(riders, "bSub_8_H_L2.txt")
# hist.fullHistComp(riders, "bSub_8-8_HS_L2.txt", channels=2)

# cv2.imshow("c", riders[0].bgCustom)
# cv2.imshow("ca", riders[0].bgBack)
# cv2.waitKey(0)

# for rider in riders:
#     videoParsing.saveVideoBGSub(rider)

# ----------------------------------------------------------------------------------------   FRAMES   -----------------------------------------------------------------------------------------------------------------------------

# # print(len(riders[1].frameAndMasksCustom))
#preprocess.checkMasks(riders[1].frameAndMasksCustom)
# preprocess.checkMasks(riders[1].frameAndMasksBack)
# # # # # print(len(riders[1].frameAndMasksBack))
# # # preprocess.checkMasks(riders[1].frameAndMasksBack)

# # cv2.imshow("test", zero.frameAndMasksCustomFull[0][0])
# # cv2.waitKey(0)

#  --------------------------------------------------------------------------------------   HISTOGRAMS   --------------------------------------------------------------------------------------------------------------------------

# a = pickle.load(open("./files/pickles/RIDER2.p", "rb"))
# b = pickle.load(open("./files/pickles/RIDER1.p", "rb"))
# preprocess.checkMasks(a.frameAndMasksCustom)
# preprocess.checkMasks(b.frameAndMasksCustom)
# print(a.bgBack.shape)
# print(a.bgCustom.shape)

# print(riders[1].customHist1D)

# figA = plt.figure()
# axA = figA.add_subplot()

# axA.plot(riders[1].backHist1D, color="black")
# axA.plot(riders[1].customHist1D, color="r")
# axA.plot(riders[0].customHist1D, color="b")
# plt.show()

# -------------------------------------------------------------------------------------------  SALIENCY  --------------------------------------------------------------------------------------------------------------------------

# f, m = riders[1].frameAndMasksBack[20]
# image = cv2.bitwise_and(f, f, mask=m)
# hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# saliency = cv2.saliency.StaticSaliencyFineGrained_create()
# (success, saliencyMap) = saliency.computeSaliency(image)
# saliencyMap = (saliencyMap * 255).astype("uint8")
# threshMap = cv2.threshold(saliencyMap.astype("uint8"), 200, 255, cv2.THRESH_BINARY)[1]

# backThresh = cv2.inRange(hsv_img, (0, 150, 150), (180, 255, 255))
# backCut = cv2.bitwise_and(image, image, mask=backThresh)
# backHist = hist.compute1DHist(f, mask=backThresh, normalize="density")

# print("back: ", backHist)

# cv2.imshow("Image", image)
# # cv2.imshow("Output", saliencyMap)
# # cv2.imshow("Thresh", threshMap)
# cv2.imshow("S", backCut)

# cF, cM = riders[1].frameAndMasksCustom[15]
# customImage = cv2.bitwise_and(cF, cF, mask=cM)
# customHsv_img = cv2.cvtColor(customImage, cv2.COLOR_BGR2HSV)

# customThresh = cv2.inRange(customHsv_img, (0, 100, 100), (180, 255, 255))
# customCut = cv2.bitwise_and(customImage, customImage, mask=customThresh)
# customHist = hist.compute1DHist(cF, mask=customThresh, normalize="density")

# cv2.imshow("Image C", image)
# cv2.imshow("S C", customCut)
# # cv2.waitKey(0)

# decoy, decoyMask = riders[3].frameAndMasksCustom[15]
# decoyImage = cv2.bitwise_and(decoy, decoy, mask=decoyMask) 
# decoyHsv_img = cv2.cvtColor(decoyImage, cv2.COLOR_BGR2HSV)

# decoyThresh = cv2.inRange(decoyHsv_img, (0, 100, 100), (180, 255, 255))
# decoyCut = cv2.bitwise_and(decoy, decoy, mask=decoyThresh)
# decoyHist = hist.compute1DHist(decoy, mask=decoyThresh, normalize="density")

# cv2.imshow("decoy", decoy)
# cv2.imshow("decoy c", decoyCut)
# cv2.waitKey(0)

# result = hist.compareHistCV(hist.softMaxHist(backHist), hist.softMaxHist(customHist), cv2.HISTCMP_BHATTACHARYYA)
# result2 = hist.compareHistCV(hist.softMaxHist(backHist), hist.softMaxHist(decoyHist), cv2.HISTCMP_BHATTACHARYYA)
# print(result, result2)

# --------------------------------------------------------- SPLIT MASK ------------------------------------------------

# f, m = riders[0].frameAndMasksCustom[15]
# m = cv2.threshold(m, 200, 255, cv2.THRESH_BINARY)[1]
# top, bottom = segmentation.cutMask(m, mod="h")

# x, y, w, h = segmentation.maskBoundingBox(m)
# m = cv2.rectangle(m, (x, y), (x+w, y+h), (255,0,0), 2)

# cv2.imshow("original", m)
# cv2.imshow("top", top)
# cv2.imshow("bottom", bottom)
# cv2.waitKey(0)


# -------------------------------------------------- TEST SINGLE FRAME --------------------------------------

# r = riders[0]

# r.singleFrameCancelletto()

# top, bottom = segmentation.cutMask(r.maxMaskBack, mod="v", inverse=False, dim=6)
# helmet = cv2.bitwise_and(r.maxFrameBack,r.maxFrameBack, mask=bottom)

# cv2.imshow("original", r.maxMaskBack)
# cv2.imshow("cut", helmet)
# cv2.imshow("top", top)
# cv2.imshow("bottom", bottom)
# cv2.waitKey(0)

# cv2.imshow("f", r.maxFrameBack)
# cv2.imshow("m", r.maxMaskBack)
# cv2.waitKey(0)

for rider in riders:
    rider.singleFrameCancelletto()
    rider.singleFrameCustom()
    rider.collectMaxHistBack(normalization="density")
    rider.collectMaxHistCustom(normalization="density")



hist.fullHistCompHelmet(riders, "helmet_bSub_8_HS_density.txt")