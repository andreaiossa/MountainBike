import sys
import numpy as np
from matplotlib import pyplot as plt
from tabulate import tabulate
from sklearn.preprocessing import MinMaxScaler
from components.utils import *
from components.hist import *

def gaussian(times):
   
    times = np.array(times)
    mean = times.mean()
    std = times.std()
    gaus = norm(mean, std)

    return gaus


def fullHistComp(riders, fileName, name, channels=1,show=False, position=False):
    tables = []
    headers = []
    times = []
    for rider in riders:
        headers.append(rider.name)
        times.append(rider.times[name][0])
    
    gaus = gaussian(times)

    for metricTuple in metrics:
        metric = metricTuple[0]
        mod = metricTuple[1]
        metricName = metricTuple[2]
        fun = metricTuple[3]
        softmax = False
        if len(metricTuple) == 5:
            softmax = True
        scoreIn1 = 0
        scoreIn3 = 0
        scoreIn5 = 0
        riderIndex = 0
        table = []
        for rider in riders:
            riderIndex += 1
            tmpRow = []
            row = [rider.name]
            shouldMatch = None
            results = []
            riderHists = rider.squashHists
            riderFeature = rider.resNetFeatures
            refIndex = 0
            for ref in riders:
                refIndex += 1
                refHists = ref.squashHists
                refFeature = ref.resNetFeatures
                if channels == 1:
                    refHist = np.float32(refHists["back"][0])
                    riderHist = np.float32(riderHists[name][0])
                if channels == 2:
                    refHist = np.float32(refHists["back"][1])
                    riderHist = np.float32(riderHists[name][1])               
                if channels == "feature":
                    refHist = np.float32(refFeature["back"])
                    riderHist = np.float32(riderFeature[name])

                if show:
                    fig1 = plt.figure(f"REF: {ref.name}")
                    displayHist(refHist, fig1, mod=1)
                    plt.show()
                    fig1 = plt.figure(rider.name)
                    displayHist(riderHist, fig1, mod=1)
                    plt.show()

                if softmax:
                    refHist = softMaxHist(refHist)
                    riderHist = softMaxHist(riderHist)
                if fun == "CV":
                    result = compareHistCV(riderHist, refHist, metric)
                elif fun == "PY":
                    result = compareHistPY(riderHist, refHist, metric)
                if position:
                    result = minMax(result)
                    weight = gaus.pdf(ref.times[name])
                    result = result*weight

                results.append((rider.name, result))
                if rider.name == ref.name:
                    shouldMatch = result

                tmpRow.append(result)

            sortedResults = sorted(results, key=lambda x: x[1]) if mod == "min" else sorted(results, reverse=True, key=lambda x: x[1])
            sortedRiders = list(map(lambda x: x[0], sortedResults))
            best = sortedResults[0][1]
            refPosition = sortedRiders.index(ref.name)

            scoreIn1 = scoreIn1 + 1 if refPosition == 0 else scoreIn1
            scoreIn3 = scoreIn3 + 1 if refPosition <= 2 else scoreIn3
            scoreIn5 = scoreIn5 + 1 if refPosition <= 5 else scoreIn5

            for r in tmpRow:
                if r == shouldMatch and r == best:
                    row.append(f"{bcolors.OKBLUE}{r}{bcolors.ENDC}")
                elif r == best:
                    row.append(f"{bcolors.OKCYAN}{r}{bcolors.ENDC}")
                else:
                    row.append(r)
            table.append(row)

        table.append([f"{bcolors.OKGREEN}{metricName}", f"Score in 1: {scoreIn1}/10", f"Score in 3: {scoreIn3}/10", f"Score in 5: {scoreIn5}/10 " f"better is {mod}{bcolors.ENDC}"])
        tables.append(table)

    original_stdout = sys.stdout
    with open(fileName, 'w') as f:
        sys.stdout = f
        for tab in tables:
            print(tabulate(tab, headers=headers))
            print("\n")
        sys.stdout = original_stdout

def fullHistCompHelmet(riders, fileName, name, coeff=0.5, position=False):
    tables = []
    headers = []
    times = []
    for rider in riders:
        headers.append(rider.name)
        times.append(rider.times[name][0])
    
    gaus = gaussian(times)

    for metricTuple in metrics:
        metric = metricTuple[0]
        mod = metricTuple[1]
        metricName = metricTuple[2]
        fun = metricTuple[3]
        softmax = False
        if len(metricTuple) == 5:
            softmax = True
        scoreIn1 = 0
        scoreIn3 = 0
        scoreIn5 = 0
        riderIndex = 0
        table = []
        for rider in riders:
            riderIndex += 1
            tmpRow = []
            row = [rider.name]
            shouldMatch = None
            results = []
            riderHists = rider.maxHists
            refIndex = 0
            for ref in riders:
                refIndex += 1
                refHists = ref.maxHists
                refHistHelmet = np.float32(refHists["back"][0])
                refHistBody = np.float32(refHists["back"][1])
                riderHistHelmet = np.float32(riderHists[name][0])
                riderHistBody = np.float32(riderHists[name][1])

                if softmax:
                    refHistHelmet = softMaxHist(refHistHelmet)
                    refHistBody = softMaxHist(refHistBody)
                    riderHistHelmet = softMaxHist(riderHistHelmet)
                    riderHistBody = softMaxHist(riderHistBody)
                if fun == "CV":
                    resultHelmet = compareHistCV(riderHistHelmet, refHistHelmet, metric)
                    resultBody = compareHistCV(riderHistBody, refHistBody, metric)
                    result = (1-coeff)*resultHelmet + coeff*resultBody
                elif fun == "PY":
                    resultHelmet = compareHistPY(riderHistHelmet, refHistHelmet, metric)
                    resultBody = compareHistPY(riderHistBody, refHistBody, metric)
                    result = (1-coeff)*resultHelmet + coeff*resultBody
                if position:
                    result = minMax(result)
                    weight = gaus.pdf(ref.times[name])
                    result = result*weight
                
                results.append((rider.name, result))
                if rider.name == ref.name:
                    shouldMatch = result

                tmpRow.append(result)

            sortedResults = sorted(results, key=lambda x: x[1]) if mod == "min" else sorted(results, reverse=True, key=lambda x: x[1])
            sortedRiders = list(map(lambda x: x[0], sortedResults))
            best = sortedResults[0][1]
            refPosition = sortedRiders.index(ref.name)

            scoreIn1 = scoreIn1 + 1 if refPosition == 0 else scoreIn1
            scoreIn3 = scoreIn3 + 1 if refPosition <= 2 else scoreIn3
            scoreIn5 = scoreIn5 + 1 if refPosition <= 5 else scoreIn5

            for r in tmpRow:
                if r == shouldMatch and r == best:
                    row.append(f"{bcolors.OKBLUE}{r}{bcolors.ENDC}")
                elif r == best:
                    row.append(f"{bcolors.OKCYAN}{r}{bcolors.ENDC}")
                else:
                    row.append(r)
            table.append(row)

        table.append([f"{bcolors.OKGREEN}{metricName}", f"Score in 1: {scoreIn1}/10", f"Score in 3: {scoreIn3}/10", f"Score in 5: {scoreIn5}/10 " f"better is {mod}{bcolors.ENDC}"])
        tables.append(table)

    original_stdout = sys.stdout
    with open(fileName, 'w') as f:
        sys.stdout = f
        for tab in tables:
            print(tabulate(tab, headers=headers))
            print("\n")
        sys.stdout = original_stdout