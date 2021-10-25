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

    def collectMasks(self, video=None, mod="detectron"):
        if mod == "detectron":
            predictor = segmentation.instantiatePredictor()
            if not video:
                self.frameAndMasksBack = videoParsing.detectronOnVideo(self.backVid, predictor, refine=True, verbose=True)
                self.frameAndMasksFront = videoParsing.detectronOnVideo(self.frontVid, predictor, refine=True, verbose=True)
            else:
                self.frameAndMasksCustom, self.frameAndMasksCustomFull = videoParsing.detectronOnVideo(video, predictor, refine=True, verbose=True)
        if mod == "bSub":
            if not video:
                self.frameAndMasksBack = videoParsing.backgroundSub(self.backVid, 0, 100000, 80, filterPerc=True)
                self.frameAndMasksFront = videoParsing.backgroundSub(self.frontVid, 0, 100000, 80, filterPerc=True)
            else:
                self.frameAndMasksCustom = videoParsing.backgroundSub(video, 0, 100000, 80, filterPerc=True)

    def collectHists(self, mod="identification"):
        if mod == "identification":
            self.frontHists1D = []
            self.frontHists2D = []
            self.backHists1D = []
            self.backHists2D = []
            for frame, mask in self.frameAndMasksFront:
                hist1D = hist.compute1DHist(frame, mask=mask, normalize=True)
                hist2D = hist.compute2DHist(frame, mask=mask, normalize=True)

                self.frontHists1D.append(hist1D[0])
                self.frontHists2D.append(hist2D)

            for frame, mask in self.frameAndMasksBack:
                hist1D = hist.compute1DHist(frame, mask=mask, normalize=True)
                hist2D = hist.compute2DHist(frame, mask=mask, normalize=True)

                self.backHists1D.append(hist1D[0])
                self.backHists2D.append(hist2D)

        if mod == "custom":
            self.customHists1D = []
            self.customHists2D = []
            for frame, mask in self.frameAndMasksCustom:
                hist1D = hist.compute1DHist(frame, mask=mask, normalize=True)
                hist2D = hist.compute2DHist(frame, mask=mask, normalize=True)

                self.customHists1D.append(hist1D[0])
                self.customHists2D.append(hist2D)

        if mod == "noMask":
            woods = self.frameAndMasksCustomFull[0][0]
            histWoods = hist.compute2DHist(woods)
            pixels = histWoods.sum()
            self.customHists1D = []
            self.customHists2D = []
            for frame, mask in self.frameAndMasksCustom:
                hist1D = hist.compute1DHist(frame, mask=None, normalize=True)
                hist2D = hist.compute2DHist(frame, mask=None, normalize=True, difference=histWoods, pixels=pixels)

                self.customHists1D.append(hist1D)
                self.customHists2D.append(hist2D)

    def squashHist(self, mod="median", channels=1):

        self.frontHist1D = hist.squashHists(self.frontHists1D, mod)
        self.backHist1D = hist.squashHists(self.backHists1D, mod)
        self.customHist1D = hist.squashHists(self.customHists1D, mod)

        self.frontHist2D = hist.squashHists(self.frontHists2D, mod)
        self.backHist2D = hist.squashHists(self.backHists2D, mod)
        self.customHist2D = hist.squashHists(self.customHists2D, mod)


RIDERfolder = "./files/RIDERS/"
picklesFolder = "./files/pickles/"


def collectRiders():
    print(f"{utils.bcolors.presetINFO} extracting pickles...")
    riders = []
    for file in os.listdir(picklesFolder):
        rider = pickle.load(open(picklesFolder + file, "rb"))
        riders.append(rider)
    print(f"{utils.bcolors.presetINFO} Done")
    return riders


def collectRidersFolders():
    RIDERS = []
    for root, dirs, files in os.walk(RIDERfolder):
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

        pickle.dump(rider, open(picklesFolder + f"{rider.name}.p", "wb"))


def updateRider(RIDER):
    pickle.dump(RIDER, open(picklesFolder + f"{RIDER.name}.p", "wb"))


def checkMasks(frameAndMask):
    for frame, mask in frameAndMask:
        if not isinstance(mask, bool):
            cut = cv2.bitwise_and(frame, frame, mask=mask)
            utils.showImgs([frame, mask, cut])
        else:
            utils.showImgs([frame])
        cv2.destroyAllWindows()