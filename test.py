import time
from cv2 import cv2
import numpy as np
from matplotlib import pyplot as plt
from numpy.lib import utils
from scipy import stats
import segmentation
import yolo
import hist
import videoParsing
import utils
import features
import os
import glob
import preprocess
import pickle
import sys
from tabulate import tabulate

riders = preprocess.collectRiders()
riders =  sorted(riders, key= lambda x: int(x.name.split("RIDER")[1]))

'''
FULL PIPELINE TEST
'''

for rider in riders:
    # rider.collectHists(mod="identification")
    # rider.collectHists(mod="custom")
    rider.squashHist(mod="mean")
    # preprocess.updateRider(rider)
print(riders[0].customHists1D[0])

# hist.fullHistComp(riders, fileName="bSub_8_H2.txt")


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