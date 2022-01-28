import sys
import cv2
import time
import pickle
import numpy as np
from PIL import Image
from scipy.spatial import distance
from matplotlib import pyplot as plt
from tabulate import tabulate
from caffe.testCaffe import testCaffe
from pipeline import fullHistComp
from riderClass import *
from videoParsing import *
from components.hist import *
from components.utils import *
from components.features import *
from components.segmentation import *
from caffe.caffe2py.caffenet import *


from sklearn.preprocessing import MinMaxScaler


### CREATE EMPYT RIDERS FROM FOLDER

# RIDERS = preprocess.collectRidersFolders()
# preprocess.processRider(RIDERS)

### IMPORT RIDERS PICKLES AND PROCESS

# riders = preprocess.collectRiders()
# riders = sorted(riders, key=lambda x: int(x.name.split("RIDER")[1]))

# model = features.resNetIstantiateModel()

# for rider in riders:

#     # rider.processFiles(customFile="jump")
#     # rider.collectMasksCancelletto()
#     # rider.collectMasksVid()
#     # rider.collectHistsCancelletto(mod="standard", normalization=cv2.NORM_L2)
#     # rider.collectHistsVid(mod="standard", normalization=cv2.NORM_L2)
#     # rider.squashHist(mod="median")
#     rider.singleFrameCancelletto()
#     rider.singleFrameCustom()
#     rider.collectResNetFeatures(model)
#     preprocess.updateRider(rider)

# # --------------------------------------------------------------------------------------- COMPARISON ------------------------------------------------------------------------------------------------------------------------------

# hist.fullHistComp(riders, "features.txt", channels="feature")
# model = resNetIstantiateModel()

import torch
import torch.nn as nn
from torchvision import models
from torchsummary import summary
import numpy as np
from scipy.stats import *
from matplotlib import pyplot as plt

# a = [x for x in range(-9, 10)]
# a = np.array(a)
# print(a.mean())
# print(a.std())
# b = norm(a.mean(), a.std())

# plt.plot(a, b.pdf(a),
#        'r-', lw=5, alpha=0.6, label='norm pdf')

# plt.show()


# ROIBackgroundSub("./files/rawVideos/back3.MP4", 200, -1, 65, "back", skip=True)


# Define the ResNet50-based Model
# class ft_net(nn.Module):
#     def __init__(self, class_num = 751):
#         super(ft_net, self).__init__()
#         #load the model
#         model_ft = models.resnet50(pretrained=True) 
#         # change avg pooling to global pooling
#         # model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
#         self.model = model_ft

#     def forward(self, x):
#         x = self.model.conv1(x)
#         x = self.model.bn1(x)
#         x = self.model.relu(x)
#         x = self.model.maxpool(x)
#         x = self.model.layer1(x)
#         x = self.model.layer2(x)
#         x = self.model.layer3(x)
#         x = self.model.layer4(x)
#         x = self.model.avgpool(x)
#         x = torch.squeeze(x)
#         x = self.classifier(x) #use our classifier.
#         return x
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)
# modelA = models.resnet50(pretrained=False)
# modelA.to(device)
# modelB = resNetIstantiateModel()
# modelB.to(device)
# summary(modelB,(3,224,224))

# hist.fullHistComp(riders, "bSub_8-8_HS_L2.txt", channels=2)

# cv2.imshow("c", riders[0].bgCustom)
# cv2.imshow("ca", riders[0].bgBack)
# cv2.waitKey(0)

# for rider in riders:
#     videoParsing.saveVideoBGSub(rider)

# ----------------------------------------------------------------------------------------   FRAMES   -----------------------------------------------------------------------------------------------------------------------------

# # print(len(riders[1].frameAndMasksCustom))
#preprocess.checkMasks(riders[1].frameAndMasksCustom)
# preprocess.checkMasks(riders[1].frameAndMasksBack)
# # # # # print(len(riders[1].frameAndMasksBack))
# # # preprocess.checkMasks(riders[1].frameAndMasksBack)

# # cv2.imshow("test", zero.frameAndMasksCustomFull[0][0])
# # cv2.waitKey(0)

#  --------------------------------------------------------------------------------------   HISTOGRAMS   --------------------------------------------------------------------------------------------------------------------------

# a = pickle.load(open("./files/pickles/RIDER2.p", "rb"))
# b = pickle.load(open("./files/pickles/RIDER1.p", "rb"))
# preprocess.checkMasks(a.frameAndMasksCustom)
# preprocess.checkMasks(b.frameAndMasksCustom)
# print(a.bgBack.shape)
# print(a.bgCustom.shape)

# print(riders[1].customHist1D)

# figA = plt.figure()
# axA = figA.add_subplot()

# axA.plot(riders[1].backHist1D, color="black")
# axA.plot(riders[1].customHist1D, color="r")
# axA.plot(riders[0].customHist1D, color="b")
# plt.show()

# -------------------------------------------------------------------------------------------  SALIENCY  --------------------------------------------------------------------------------------------------------------------------

# f, m = riders[1].frameAndMasksBack[20]
# image = cv2.bitwise_and(f, f, mask=m)
# hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# saliency = cv2.saliency.StaticSaliencyFineGrained_create()
# (success, saliencyMap) = saliency.computeSaliency(image)
# saliencyMap = (saliencyMap * 255).astype("uint8")
# threshMap = cv2.threshold(saliencyMap.astype("uint8"), 200, 255, cv2.THRESH_BINARY)[1]

# backThresh = cv2.inRange(hsv_img, (0, 150, 150), (180, 255, 255))
# backCut = cv2.bitwise_and(image, image, mask=backThresh)
# backHist = hist.compute1DHist(f, mask=backThresh, normalize="density")

# print("back: ", backHist)

# cv2.imshow("Image", image)
# # cv2.imshow("Output", saliencyMap)
# # cv2.imshow("Thresh", threshMap)
# cv2.imshow("S", backCut)

# cF, cM = riders[1].frameAndMasksCustom[15]
# customImage = cv2.bitwise_and(cF, cF, mask=cM)
# customHsv_img = cv2.cvtColor(customImage, cv2.COLOR_BGR2HSV)

# customThresh = cv2.inRange(customHsv_img, (0, 100, 100), (180, 255, 255))
# customCut = cv2.bitwise_and(customImage, customImage, mask=customThresh)
# customHist = hist.compute1DHist(cF, mask=customThresh, normalize="density")

# cv2.imshow("Image C", image)
# cv2.imshow("S C", customCut)
# # cv2.waitKey(0)

# decoy, decoyMask = riders[3].frameAndMasksCustom[15]
# decoyImage = cv2.bitwise_and(decoy, decoy, mask=decoyMask) 
# decoyHsv_img = cv2.cvtColor(decoyImage, cv2.COLOR_BGR2HSV)

# decoyThresh = cv2.inRange(decoyHsv_img, (0, 100, 100), (180, 255, 255))
# decoyCut = cv2.bitwise_and(decoy, decoy, mask=decoyThresh)
# decoyHist = hist.compute1DHist(decoy, mask=decoyThresh, normalize="density")

# cv2.imshow("decoy", decoy)
# cv2.imshow("decoy c", decoyCut)
# cv2.waitKey(0)

# result = hist.compareHistCV(hist.softMaxHist(backHist), hist.softMaxHist(customHist), cv2.HISTCMP_BHATTACHARYYA)
# result2 = hist.compareHistCV(hist.softMaxHist(backHist), hist.softMaxHist(decoyHist), cv2.HISTCMP_BHATTACHARYYA)
# print(result, result2)

# --------------------------------------------------------- SPLIT MASK ------------------------------------------------

# f, m = riders[0].frameAndMasksCustom[15]
# m = cv2.threshold(m, 200, 255, cv2.THRESH_BINARY)[1]
# top, bottom = segmentation.cutMask(m, mod="h")

# x, y, w, h = segmentation.maskBoundingBox(m)
# crop = m[y:y+h, x:x+w]
# m = cv2.rectangle(m, (x, y), (x+w, y+h), (255,0,0), 2)


# cv2.imshow("original", m)
# cv2.imshow("top", top)
# cv2.imshow("bottom", bottom)
# cv2.imshow("crop", crop)
# cv2.waitKey(0)


# -------------------------------------------------- TEST SINGLE FRAME --------------------------------------

# r = riders[0]

# r.singleFrameCancelletto()
# r.singleFrameCustom()

# top, bottom = segmentation.cutMask(r.maxMaskBack, mod="v", inverse=False, dim=6)
# helmet = cv2.bitwise_and(r.maxFrameBack,r.maxFrameBack, mask=r.maxMaskBack)

# # cv2.imshow("original", r.maxMaskBack)
# # cv2.imshow("cut", helmet)
# # cv2.imshow("top", top)
# # cv2.imshow("bottom", bottom)
# # cv2.waitKey(0)

# cv2.imshow("f", r.maxFrameBack)
# cv2.imshow("c", r.maxFrameCustom)
# cv2.imshow("g", r.maxFrameCustomMask)
# # cv2.imshow("m", r.maxMaskBack)
# cv2.waitKey(0)

# # for rider in riders:
# #     rider.singleFrameCancelletto()
# #     rider.singleFrameCustom()
# #     rider.collectMaxHistBack(normalization="density")
# #     rider.collectMaxHistCustom(normalization="density")



# # hist.fullHistComp(riders, "helmet_bSub_8_HS_density.txt", channels=2)


# -------------------------------------------------- TEST CAFFE --------------------------------------


# r = riders[0]

# r.singleFrameCancelletto()
# r.singleFrameCustom()



# A = segmentation.cropImageBbox(r.maxFrameBack, r.maxFrameBackMask)
# B = segmentation.cropImageBbox(r.maxFrameCustom, r.maxFrameCustomMask)
# # cv2.imshow("b", r.maxFrameBack)
# # cv2.imshow("c", r.maxFrameCustom)
# # cv2.imshow("m", r.maxFrameCustomMask)
# # cv2.imshow("A", A)
# # cv2.imshow("B", B)
# # cv2.waitKey(0)

# cv2.imwrite("A.jpg", A)
# cv2.imwrite("B.jpg", B)

# a = Image.open("A.jpg")
# b = Image.open("B.jpg")

# model = CaffeNet('caffe/models/ResNet_50/ResNet_50_test.prototxt')
# model.load_state_dict(torch.load('caffe/test.pt'))

# featureA = testCaffe(a, model)
# featureB = testCaffe(b, model)

# dst = distance.euclidean(featureA, featureB)
# print(dst)


# -------------------------------
'''
riders = collectRidersFolders()
riders = sorted(riders, key= lambda r: int(r.name.split("RIDER")[1]))

for r in riders:
    r.processFiles()
    r.collectFM("front", 80, (400,400), 3)
    r.collectFM("turn", 80, (400,400), 3)
    r.framesAndMasks["front"].collectHistsCancelletto()
    r.framesAndMasks["turn"].collectHistsCancelletto()
    r.squashHist("front")
    r.squashHist("turn")
updateRiders(riders)
'''

# riders = collectRiders()
# riders = sorted(riders, key= lambda r: int(r.name.split("RIDER")[1]))

# fullHistComp(riders, "bb.txt", "turn", "front", channels=2, position=True)

# from pipeline import gaussian

# times = []
# for rider in riders:    
#     times.append(rider.times["turn"][0] + rider.times["front"][1])

# times = np.array(times)
# print(times)
# print(f"MEAN= {times.mean()}, STD = {times.std()}")
# times = zScore(times)
# print(f"MEAN= {times.mean()}, STD = {times.std()}")

# gaus = gaussian(times)

# print(times)


# # print(f"aa {times[0]}")
# print(gaus.pdf(times[3]))
# # print(gaus.pdf(111913))


#----------

#first 8 riders are back1   (52350 + 81990 ) 134340
#next 2 are back2   (81990)


# riders = collectRidersFolders()
# riders = sorted(riders, key= lambda r: int(r.name.split("RIDER")[1]))

# # riders = riders[0:8]

# print(riders[-1].name)


# for r in riders:

#     r.processFiles()
#     r.collectFM("back", 80, (400,400), 3)
#     r.collectFM("para", 80, (400,400), 3)
#     r.framesAndMasks["back"].collectHistsCancelletto()
#     r.framesAndMasks["para"].collectHistsCancelletto()
#     r.squashHist("back")
#     r.squashHist("para")
# updateRiders(riders)


riders = collectRiders()
riders = sorted(riders, key= lambda r: int(r.name.split("RIDER")[1]))
riders = riders[0:8]
fullHistComp(riders, "para_HS_64_32_n3.txt", "para", "back", channels=1, position=True)

