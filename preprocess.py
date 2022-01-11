import os
from cv2 import cv2
import segmentation
import videoParsing
import hist
import utils
import pickle
import numpy as np
from Cc.testCaffe import testCaffe
from PIL import Image


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

    def collectMasksCancelletto(self):
        self.frameAndMasksBack, self.bgBack = videoParsing.backgroundSub(self.backVid, 0, 100000, 80, filterPerc=True)
        # self.frameAndMasksFront, self.bgFront = videoParsing.backgroundSub(self.frontVid, 0, 100000, 80, filterPerc=True)

    def collectMasksVid(self):
        self.frameAndMasksCustom, self.bgCustom = videoParsing.backgroundSub(self.customVid, 0, 100000, 80, filterPerc=True)

    def collectHistsCancelletto(self, mod="standard", normalization=cv2.NORM_MINMAX):
        # self.frontHists1D = []
        # self.frontHists2D = []
        self.backHists1D = []
        self.backHists2D = []
        if mod == "standard":
            # for frame, mask in self.frameAndMasksFront:
            #     hist1D = hist.compute1DHist(frame, mask=mask, normalize=normalization)
            #     hist2D = hist.compute2DHist(frame, mask=mask, normalize=normalization)

            #     self.frontHists1D.append(hist1D[0])
            #     self.frontHists2D.append(hist2D)

            for frame, mask in self.frameAndMasksBack:
                hist1D = hist.compute1DHist(frame, mask=mask, normalize=normalization)
                hist2D = hist.compute2DHist(frame, mask=mask, normalize=normalization)

                self.backHists1D.append(hist1D[0])
                self.backHists2D.append(hist2D)
        if mod == "diff":
            # for frame, mask in self.frameAndMasksFront:
            #     hist1D = hist.compute1DHist(frame, mask=mask, normalize=normalization)
            #     histBg1D = hist.compute1DHist(self.bgFront, mask=mask, normalize=normalization)

            #     H = hist.diffHist(hist1D[0], histBg1D[0])

            #     self.frontHists1D.append(H)

            for frame, mask in self.frameAndMasksBack:
                hist1D = hist.compute1DHist(frame, mask=mask)
                histBg1D = hist.compute1DHist(self.bgBack, mask=mask)
                hist2D = hist.compute2DHist(frame, mask=mask)
                histBg2D = hist.compute2DHist(self.bgBack, mask=mask)

                H = hist.diffHist(hist1D[0], histBg1D[0])
                HS = hist.diffHist(hist2D, histBg2D)

                H = hist.histNormalize(H, normalize=normalization)
                HS = hist.histNormalize(HS, normalize=normalization)

                self.backHists1D.append(H)
                self.backHists2D.append(HS)

    def collectHistsVid(self, mod="standard", normalization=cv2.NORM_MINMAX):
        self.customHists1D = []
        self.customHists2D = []
        if mod == "standard":
            for frame, mask in self.frameAndMasksCustom:
                hist1D = hist.compute1DHist(frame, mask=mask, normalize=normalization)
                hist2D = hist.compute2DHist(frame, mask=mask, normalize=normalization)

                self.customHists1D.append(hist1D)
                self.customHists2D.append(hist2D)
        if mod == "diff":
            for frame, mask in self.frameAndMasksCustom:
                hist1D = hist.compute1DHist(frame, mask=mask)
                hist2D = hist.compute2DHist(frame, mask=mask)
                histBg1D = hist.compute1DHist(self.bgCustom)
                histBg2D = hist.compute2DHist(self.bgCustom)

                H = hist.diffHist(hist1D, histBg1D)
                HS = hist.diffHist(hist2D, histBg2D)

                H = hist.histNormalize(H, normalize=normalization)
                HS = hist.histNormalize(HS, normalize=normalization)

                self.customHists1D.append(H)
                self.customHists2D.append(HS)

    def squashHist(self, mod="median"):

        # self.frontHist1D = hist.squashHists(self.frontHists1D, mod)
        self.backHist1D = hist.squashHists(self.backHists1D, mod)
        self.customHist1D = hist.squashHists(self.customHists1D, mod)

        # self.frontHist2D = hist.squashHists(self.frontHists2D, mod)
        self.backHist2D = hist.squashHists(self.backHists2D, mod)
        self.customHist2D = hist.squashHists(self.customHists2D, mod)

    def singleFrameCancelletto(self):
        maxDiff = 0
        for f, m in self.frameAndMasksBack:
            cut = cv2.bitwise_and(f, f, mask=m)
            percentageDiff = (np.count_nonzero(cut) * 100) / cut.size
            if percentageDiff >= maxDiff:
                self.maxFrameBack = f
                maxFrameBackMask = m
                self.maxFrameBackMask = cv2.threshold(maxFrameBackMask, 200, 255, cv2.THRESH_BINARY)[1]
                m = cv2.threshold(m, 200, 255, cv2.THRESH_BINARY)[1]
                self.maxMaskBack = m
                maxDiff = percentageDiff

    def singleFrameCustom(self):
        maxDiff = 0
        for f, m in self.frameAndMasksCustom:
            cut = cv2.bitwise_and(f, f, mask=m)
            percentageDiff = (np.count_nonzero(cut) * 100) / cut.size
            if percentageDiff >= maxDiff:
                self.maxFrameCustom = f
                maxFrameCustomMask = m
                self.maxFrameCustomMask = cv2.threshold(maxFrameCustomMask, 200, 255, cv2.THRESH_BINARY)[1]
                m = cv2.threshold(m, 200, 255, cv2.THRESH_BINARY)[1]
                self.maxMaskCustom = m
                maxDiff = percentageDiff

    def collectMaxHistBack(self, normalization):
        self.topBack, self.bottomBack = segmentation.cutMask(self.maxMaskBack, mod="v", inverse=False, dim=6)
        self.helmetHistBack2D = hist.compute2DHist(self.maxFrameBack, mask=self.topBack, normalize=normalization)
        self.bottomHistBack2D = hist.compute2DHist(self.maxFrameBack, mask=self.bottomBack, normalize=normalization)


    def collectMaxHistCustom(self, normalization):
        self.topCustom, self.bottomCustom = segmentation.cutMask(self.maxMaskCustom, mod="v", inverse=True, dim=6)
        self.helmetHistCustom2D = hist.compute2DHist(self.maxFrameCustom, mask=self.topCustom, normalize=normalization)
        self.bottomHistCustom2D = hist.compute2DHist(self.maxFrameCustom, mask=self.bottomCustom, normalize=normalization)

    def collectResNetFeatures(self, model):
        a = segmentation.cropImageBbox(self.maxFrameBack, self.maxFrameBackMask)
        b = segmentation.cropImageBbox(self.maxFrameCustom, self.maxFrameCustomMask)
        
        cv2.imwrite("files/temp/a.jpg",a)
        cv2.imwrite("files/temp/b.jpg",b)
        a = Image.open("files/temp/a.jpg")
        b = Image.open("files/temp/b.jpg")
        self.backFeatures = testCaffe(a, model)
        self.customFeatures = testCaffe(b, model)


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