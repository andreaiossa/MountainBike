import os
from cv2 import cv2
import segmentation
import videoParsing
import hist
import utils
import pickle


class rider():
    def __init__(self, name, folder):
        self.folder = folder
        self.files = os.listdir(folder)
        self.name = name
        self.frontImg = None
        self.backImg = None
        self.frontVid = None
        self.backVid = None
        self.frontHists1D = []
        self.frontHists2D = []
        self.backHists1D = []
        self.backHists2D = []
        self.frameAndMasksFront = []
        self.frameAndMasksBack = []
        self.frontHist2D = None
        self.backHist2D = None

        self.frameAndMasksCustom = []
        self.customHist2D = None
        self.customImg = None
        self.customVid = None

    def processFiles(self, customFile=None):
        if not customFile:
            for file in self.files:
                if file.find("id_front.jpg") != -1:
                    self.frontImg = cv2.imread(os.path.join(self.folder, file))
                if file.find("id_back.jpg") != -1:
                    self.backImg = cv2.imread(os.path.join(self.folder, file))
                if file.find("id_front.avi") != -1:
                    self.frontVid = os.path.join(self.folder, file)
                if file.find("id_back.avi") != -1:
                    self.backVid = os.path.join(self.folder, file)

        else:
            for file in self.files:
                if file.find(customFile + ".jpg") != -1:
                    self.customImg = cv2.imread(os.path.join(self.folder, file))
                if file.find(customFile + ".avi") != -1:
                    self.customVid = os.path.join(self.folder, file)

    def collectMasks(self, video=None):
        predictor = segmentation.instantiatePredictor()
        if not video:
            self.frameAndMasksBack = videoParsing.detectronOnVideo(self.backVid, predictor, refine=True, verbose=True)
            self.frameAndMasksFront = videoParsing.detectronOnVideo(self.frontVid, predictor, refine=True, verbose=True)
        else:
            self.frameAndMasksCustom = videoParsing.detectronOnVideo(video, predictor, refine=True, verbose=True)

    def collectHists(self, mod="identification"):
        if mod == "identification":
            for frame, mask in self.frameAndMasksFront:
                hist1D = hist.compute1DHist(frame, mask=mask, normalize=True)
                hist2D = hist.compute2DHist(frame, mask=mask, normalize=True)

                self.frontHists1D.append(hist1D)
                self.frontHists2D.append(hist2D)

            for frame, mask in self.frameAndMasksBack:
                hist1D = hist.compute1DHist(frame, mask=mask, normalize=True)
                hist2D = hist.compute2DHist(frame, mask=mask, normalize=True)

                self.backHists1D.append(hist1D)
                self.backHists2D.append(hist2D)

        if mod == "custom":
            for frame, mask in self.frameAndMasksCustom:
                hist1D = hist.compute1DHist(frame, mask=mask, normalize=True)
                hist2D = hist.compute2DHist(frame, mask=mask, normalize=True)

                self.customHists1D = []
                self.customHists2D = []
                self.customHists1D.append(hist1D)
                self.customHists2D.append(hist2D)

    def squashHist(self, mod="median", update=False):
        self.frontHist2D = hist.squashHists(self.frontHists2D, mod)
        self.backHist2D = hist.squashHists(self.backHists2D, mod)
        if update:
            self.customHist2D = hist.squashHists(self.customHists2D, mod)


folder = "./files/RIDERS"
saveFolder = "./files/pickles/"


def collectRiders():
    RIDERS = []
    for root, dirs, files in os.walk(folder):
        for dirName in dirs:
            dirPath = os.path.join(root, dirName)
            r = rider(dirName, dirPath)
            RIDERS.append(r)

    return RIDERS


def processRider(RIDERS):
    counter = 0
    tot = len(RIDERS)
    for rider in RIDERS:
        counter += 1
        print(f"{utils.bcolors.presetINFO} Processing {counter}/{tot}...")
        rider.processFiles()
        rider.collectMasks()
        rider.collectHists()
        rider.squashHist()

        pickle.dump(rider, open(saveFolder + f"{rider.name}.p", "wb"))


def updateRider(RIDER):
    RIDER.processFiles(customFile="jump")
    RIDER.collectMasks(video=RIDER.customVid)
    RIDER.collectHists(mod="custom")
    RIDER.squashHist(update=True)

    pickle.dump(RIDER, open(saveFolder + f"{RIDER.name}.p", "wb"))


def checkDetectron(frameAndMask):
    for frame, mask in frameAndMask:
        cut = cv2.bitwise_and(frame, frame, mask=mask)
        utils.showImgs([frame, mask, cut])
        cv2.destroyAllWindows()