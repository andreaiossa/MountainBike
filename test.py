import time
import cv2
import numpy as np
from matplotlib import pyplot as plt
import preprocess
import pickle
import sys
from tabulate import tabulate
import hist

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

# ### COMPARISON

# hist.fullHistComp(riders, "bSub_8_H_L2.txt")
# hist.fullHistComp(riders, "bSub_8-8_HS_L2.txt", channels=2)

cv2.imshow("c", riders[0].bgCustom)
cv2.imshow("ca", riders[0].bgBack)
cv2.waitKey(0)

### FRAMES

# # print(len(riders[1].frameAndMasksCustom))
# # preprocess.checkMasks(riders[1].frameAndMasksCustom)
# # # # # print(len(riders[1].frameAndMasksBack))
# # # preprocess.checkMasks(riders[1].frameAndMasksBack)

# # cv2.imshow("test", zero.frameAndMasksCustomFull[0][0])
# # cv2.waitKey(0)

### HISTOGRAMS

# a = pickle.load(open("./files/pickles/RIDER2.p", "rb"))
# b = pickle.load(open("./files/pickles/RIDER1.p", "rb"))
# preprocess.checkMasks(a.frameAndMasksCustom)
# preprocess.checkMasks(b.frameAndMasksCustom)
# print(a.bgBack.shape)
# print(a.bgCustom.shape)

print(riders[1].customHist1D)

figA = plt.figure()
axA = figA.add_subplot()

axA.plot(riders[1].backHist1D, color="black")
axA.plot(riders[1].customHist1D, color="r")
axA.plot(riders[0].customHist1D, color="b")
plt.show()
