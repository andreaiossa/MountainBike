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

riders =  sorted(riders, key= lambda x: int(x.name.split("RIDER")[1]))

for rider in riders:
    rider.collectHists(mod="identification")
    rider.collectHists(mod="noMask")
    rider.squashHist(mod="median", update=True)

tables = []
headers = []
for rider in riders:
    headers.append(rider.name)

for metricTuple in utils.metrics:
    metric = metricTuple[0]
    mod = metricTuple[1]
    metricName = metricTuple[2]
    fun = metricTuple[3]
    score = 0
    table = []
    for ref in riders:
        tmpRow = []
        row = [ref.name]
        minimum = None
        maximum = None
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
                if not minimum:
                    minimum = result
                    match = rider
                elif result < minimum:
                    minimum = result
                    match = rider
            if mod == "max":
                if not maximum:
                    maximum = result
                    match = rider
                elif result > maximum:
                    maximum = result
                    match = rider
            if rider.name == ref.name:
                shouldMatch = result

            tmpRow.append(result)

        best = minimum if mod == "min" else maximum
        score = score + 1 if ref.name == match.name else score
        for r in tmpRow:
            if r == shouldMatch and r == best:
                row.append(f"{utils.bcolors.OKBLUE}{r}{utils.bcolors.ENDC}")
            elif r == best:
                row.append(f"{utils.bcolors.OKCYAN}{r}{utils.bcolors.ENDC}")
            else:
                row.append(r)
        table.append(row)
        
        # matchName = match.name if match != False else "NOMATCH"
        # print(f"\t {ref.name}: Best match was {matchName} with {best}")
        # print(f"\t comparison with self was {shouldMatch }")
    table.append([f"{utils.bcolors.OKGREEN}{metricName}", f"Score: {score}/10", f"better is {mod}{utils.bcolors.ENDC}"])
    tables.append(table)
    # print(f"{utils.bcolors.presetINFO} Total score for {metricName} is {score}/{len(riders)}")

original_stdout = sys.stdout
with open('detectron_MinMax.txt', 'w') as f:
    sys.stdout = f
    for tab in tables:
        print(tabulate(tab, headers=headers))
        print("\n")
    sys.stdout = original_stdout



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