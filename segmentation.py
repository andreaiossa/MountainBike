from cv2 import cv2
import numpy as np
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
import argparse
import time
import os
from PIL import Image

im = cv2.imread("./files/immagini/back_green.jpg")
w = int(im.shape[1] / 3)
h = int(im.shape[0] / 3)
im = cv2.resize(im, (w, h))
# cv2.imshow("a", im_resized)

# cfg = get_config("Misc/scratch_mask_rcnn_R_50_FPN_3x_gn.yaml", trained=True)
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

predictor = DefaultPredictor(cfg)
outputs = predictor(im)

# print(outputs["instances"].pred_classes)
# print(outputs["instances"].pred_masks)
mask = outputs["instances"].pred_masks[0]

mask = np.array(mask.cpu(), dtype=np.uint8)
mask2 = mask
mask = Image.fromarray(mask * 255)
mask.save("mask.jpg")
mask = cv2.imread("mask.jpg", cv2.IMREAD_GRAYSCALE)
cv2.imwrite("mask.jpg", mask)

mask[mask > 0] = cv2.GC_PR_FGD
mask[mask == 0] = cv2.GC_BGD
fgModel = np.zeros((1, 65), dtype="float")
bgModel = np.zeros((1, 65), dtype="float")

roughOutput = cv2.bitwise_and(im, im, mask=mask)

(mask, bgModel, fgModel) = cv2.grabCut(im, mask, None, bgModel, fgModel, iterCount=1, mode=cv2.GC_INIT_WITH_MASK)

outputMask = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD), 0, 1)
outputMask = (outputMask * 255).astype("uint8")
roughOutput = cv2.bitwise_and(im, im, mask=mask2)
output = cv2.bitwise_and(im, im, mask=outputMask)

cv2.imshow("Rough Output", roughOutput)
cv2.imshow("Output", output)
cv2.waitKey(0)
