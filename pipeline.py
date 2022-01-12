import sys
import numpy as np
from matplotlib import pyplot as plt
from tabulate import tabulate
from components.utils import *
from components.hist import *

def fullHistComp(riders, fileName, channels=1, show=False):
    tables = []
    headers = []
    for rider in riders:
        headers.append(rider.name)

    for metricTuple in utils.metrics:
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
        table = []
        for ref in riders:
            tmpRow = []
            row = [ref.name]
            shouldMatch = None
            results = []
            for rider in riders:
                if channels == 1:
                    refHist = np.float32(ref.backHist1D)
                    riderHist = np.float32(rider.customHist1D)
                if channels == 2:
                    refHist = np.float32(ref.helmetHistBack2D)
                    riderHist = np.float32(rider.helmetHistCustom2D)
                
                if channels == "feature":
                    refHist = np.float32(ref.backFeatures)
                    riderHist = np.float32(rider.customFeatures)

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
                    row.append(f"{utils.bcolors.OKBLUE}{r}{utils.bcolors.ENDC}")
                elif r == best:
                    row.append(f"{utils.bcolors.OKCYAN}{r}{utils.bcolors.ENDC}")
                else:
                    row.append(r)
            table.append(row)

        table.append([f"{utils.bcolors.OKGREEN}{metricName}", f"Score in 1: {scoreIn1}/10", f"Score in 3: {scoreIn3}/10", f"Score in 5: {scoreIn5}/10 " f"better is {mod}{utils.bcolors.ENDC}"])
        tables.append(table)

    original_stdout = sys.stdout
    with open(fileName, 'w') as f:
        sys.stdout = f
        for tab in tables:
            print(tabulate(tab, headers=headers))
            print("\n")
        sys.stdout = original_stdout

def fullHistCompHelmet(riders, fileName, show=False):
    tables = []
    headers = []
    for rider in riders:
        headers.append(rider.name)

    for metricTuple in utils.metrics:
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
        table = []
        for ref in riders:
            tmpRow = []
            row = [ref.name]
            shouldMatch = None
            results = []
            for rider in riders:
                
                refHistHelmet = np.float32(ref.helmetHistBack2D)
                refHistBottom = np.float32(ref.bottomHistBack2D)
                riderHistHelmet = np.float32(rider.helmetHistCustom2D)
                riderHistBottom = np.float32(rider.bottomHistCustom2D)

                if softmax:
                    refHistHelmet = softMaxHist(refHistHelmet)
                    refHistBottom = softMaxHist(refHistBottom)
                    riderHistHelmet = softMaxHist(riderHistHelmet)
                    riderHistBottom = softMaxHist(riderHistBottom)
                if fun == "CV":
                    resultHelmet = compareHistCV(riderHistHelmet, refHistHelmet, metric)
                    resultBottom = compareHistCV(riderHistBottom, refHistBottom, metric)
                    result = 0.5*resultHelmet + 0.5*resultBottom
                elif fun == "PY":
                    resultHelmet = compareHistPY(riderHistHelmet, refHistHelmet, metric)
                    resultBottom = compareHistPY(riderHistBottom, refHistBottom, metric)
                    result = 0.5*resultHelmet + 0.5*resultBottom
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
                    row.append(f"{utils.bcolors.OKBLUE}{r}{utils.bcolors.ENDC}")
                elif r == best:
                    row.append(f"{utils.bcolors.OKCYAN}{r}{utils.bcolors.ENDC}")
                else:
                    row.append(r)
            table.append(row)


        table.append([f"{utils.bcolors.OKGREEN}{metricName}", f"Score in 1: {scoreIn1}/10", f"Score in 3: {scoreIn3}/10", f"Score in 5: {scoreIn5}/10 " f"better is {mod}{utils.bcolors.ENDC}"])
        tables.append(table)

    original_stdout = sys.stdout
    with open(fileName, 'w') as f:
        sys.stdout = f
        for tab in tables:
            print(tabulate(tab, headers=headers))
            print("\n")
        sys.stdout = original_stdout