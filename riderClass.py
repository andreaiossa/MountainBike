import os
import pickle
import numpy as np
from cv2 import cv2
from PIL import Image
from caffe.testCaffe import testCaffe
from components.segmentation import *
from videoParsing import *
from components.hist import *
from components.utils import *


class fm():
    def __init__(self,video, tresh,size, filterPerc):
        self.Hists1D = []
        self.Hists2D = []
        self.framesAndMasks, self.background = backgroundSub(video, 0, 100000, tresh, size=size, filterPerc=filterPerc)

    def singleFrameCancelletto(self):
        maxDiff = 0
        maxFrame = None
        maxMask = None
        for f, m in self.framesAndMasks:
            cut = cv2.bitwise_and(f, f, mask=m)
            percentageDiff = (np.count_nonzero(cut) * 100) / cut.size
            if percentageDiff >= maxDiff:
                maxFrame = f
                maxMask = m
                maxDiff = percentageDiff
        self.maxFrame = maxFrame
        self.maxMask = maxMask

    def collectHistsCancelletto(self, mod="standard", normalization=cv2.NORM_MINMAX):
        if mod == "standard":
            for frame, mask in self.framesAndMasks:
                hist1D = compute1DHist(frame, mask=mask, normalize=normalization)
                hist2D = compute2DHist(frame, mask=mask, normalize=normalization)

                self.Hists1D.append(hist1D[0])
                self.Hists2D.append(hist2D)
        if mod == "diff":
            for frame, mask in self.framesAndMasks:
                hist1D = compute1DHist(frame, mask=mask)
                histBg1D = compute1DHist(self.background, mask=mask)
                hist2D = compute2DHist(frame, mask=mask)
                histBg2D = compute2DHist(self.background, mask=mask)

                H = diffHist(hist1D[0], histBg1D[0])
                H = histNormalize(H, normalize=normalization)
                HS = diffHist(hist2D, histBg2D)
                HS = histNormalize(HS, normalize=normalization)

                self.Hists1D.append(H)
                self.Hists2D.append(HS)

class rider():
    def __init__(self, name, folder):
        self.name = name
        self.folder = folder
        self.files = os.listdir(folder)
        self.backImg = None
        self.backVid = None
        self.frontImg = None
        self.frontVid = None
        self.customImg = None
        self.customVid = None
        self.maxHists = {}
        self.squashHists = {}
        self.framesAndMasks = {}
        self.resNetFeatures = {}
        
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

    def collectFM(self, name, video, tresh, size, filterPerc):
        # ES: back, self.backVid,80, (400,400), 3
        self.framesAndMasks[name] = fm(video, tresh, size, filterPerc)

    def squashHist(self, name, mod="median"):
        
        fm = self.framesAndMasks[name]
        hists1D = fm.hists1D
        hists2D = fm.hists2D
        hist1D = squashHists(hists1D, mod)
        hist2D = squashHists(hists2D, mod)
        self.squashHists[name] = (hist1D, hist2D)

    def collectMaxHist(self, name, normalization, mod="v", inverse=False):
        #   Check that mod (vertical etc..) is suited for the fm
        fm = self.framesAndMasks[name]
        helmet, body = cutMask(fm.maxMask, mod=mod, inverse=inverse, dim=6)
        helmetHist2D = compute2DHist(fm.maxFrame, mask=helmet, normalize=normalization)
        bodyHist2D = compute2DHist(fm.maxFrame, mask=body, normalize=normalization)
        self.maxHists[name] = (helmetHist2D, bodyHist2D)

    def collectResNetFeatures(self, name, model):
        fm = self.framesAndMasks[name]
        tempImg = cropImageBbox(fm.maxFrame, fm.maxMask)
        cv2.imwrite("files/temp/tempImg.jpg",tempImg)
        Img = Image.open("files/temp/tempImg.jpg")
        features = testCaffe(Img, model)
        self.resNetFeatures[name] = features


RIDERfolder = "./files/RIDERS/"
picklesFolder = "./files/pickles/"

def collectRiders():
    """
    Load riders from pickel files 

    Returns:
        [rider]: array of rider class
    """
    print(f"{bcolors.presetINFO} extracting pickles...")
    riders = []
    for file in os.listdir(picklesFolder):
        rider = pickle.load(open(picklesFolder + file, "rb"))
        riders.append(rider)
    print(f"{bcolors.presetINFO} Done")
    return riders


def collectRidersFolders():
    """
    Parse rider folders and istantiate a new rider object from each folder

    Returns:
        [riders]: array of rider class
    """
    riders = []
    for root, dirs, files in os.walk(RIDERfolder):
        for dirName in dirs:
            dirPath = os.path.join(root, dirName)
            r = rider(dirName, dirPath)
            riders.append(r)

    return riders


def updateRiders(riders):
    """
    create new pickle object from list of rider class processing its content

    Args:
        riders ([rider]): array of rider
    """
    counter = 0
    tot = len(riders)
    for rider in riders:
        counter += 1
        print(f"{bcolors.presetINFO} Processing {counter}/{tot}...")
        rider.processFiles()

        pickle.dump(rider, open(picklesFolder + f"{rider.name}.p", "wb"))


def updateRider(rider):
    """
    Istantiate new pickle from rider class

    Args:
        rider ([type]): [description]
    """
    pickle.dump(rider, open(picklesFolder + f"{rider.name}.p", "wb"))


# NOT CORE FUNCTIONS

def checkMasks(frameAndMask):
    for frame, mask in frameAndMask:
        if not isinstance(mask, bool):
            cut = cv2.bitwise_and(frame, frame, mask=mask)
            showImgs([frame, mask, cut])
        else:
            showImgs([frame])
        cv2.destroyAllWindows()