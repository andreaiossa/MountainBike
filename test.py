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

riders = preprocess.collectRiders()
# img = cv2.imread("files/imgs/darkgreen.jpeg")
# histo = hist.compute2DHist(img)

# woods = riders[0].frameAndMasksCustomFull[0][0]
# histWoods = hist.compute2DHist(woods, normalize=True)

# fig = plt.figure("bo")
# hist.displayHist(histo, fig, mod=1)
# plt.show()

# fig = plt.figure("bo")
# hist.displayHist(histWoods, fig, mod=1)
# plt.show()
'''
FULL PIPELINE TEST
'''

for rider in riders:
    rider.collectHists(mod="identification")
    rider.collectHists(mod="noMask")
    rider.squashHist(mod="median", update=True)

for metricTuple in utils.metrics:
    metric = metricTuple[0]
    mod = metricTuple[1]
    metricName = metricTuple[2]
    fun = metricTuple[3]
    score = 0
    for ref in riders:
        minimum = 10000000
        maximum = -1000000
        match = False
        shouldMatch = None
        for rider in riders:
            refHist = np.float32(ref.backHist2D)
            riderHist = np.float32(rider.customHist2D)

            # fig1 = plt.figure(f"REF: {ref.name}")
            # hist.displayHist(refHist, fig1, mod=1)
            # plt.show()
            # fig1 = plt.figure(rider.name)
            # hist.displayHist(riderHist, fig1, mod=1)
            # plt.show()
            if fun == "CV":
                result = hist.compareHistCV(riderHist, refHist, metric)
            elif fun == "PY":
                result = hist.compareHistPY(riderHist, refHist, metric)
            if mod == "min":
                if result < minimum:
                    minimum = result
                    match = rider
            if mod == "max":
                if result > maximum:
                    maximum = result
                    match = rider
            if rider.name == ref.name:
                shouldMatch = result

        best = minimum if mod == "min" else maximum
        matchName = match.name if match != False else "NOMATCH"
        print(f"\t {ref.name}: Best match was {matchName} with {best}")
        print(f"\t comparison with self was {shouldMatch }")
        score = score + 1 if ref.name == match.name else score

    print(f"{utils.bcolors.presetINFO} Total score for {metricName} is {score}/10")
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