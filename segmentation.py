from cv2 import cv2
import numpy as np
# from detectron2 import model_zoo
# from detectron2.config import get_cfg
# from detectron2.engine import DefaultPredictor
from os.path import isfile
from PIL import Image

import utils


# def instantiatePredictor():
#     '''
#     Instantiate detectron2 Mask rcnn predictor from web
#     '''
#     print("Generating predicor...")
#     cfg = get_cfg()
#     if (isfile("./config.yaml")):
#         cfg.merge_from_file("./config.yaml")
#     else:
#         cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
#         cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
#         cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

#     with open("config.yaml", "w") as f:
#         f.write(cfg.dump())

#     predictor = DefaultPredictor(cfg)
#     print("Done")
#     return predictor


def computeSegmentationMask(image, predictor, refine=False, CocoClass=0, verbose=True):
    '''
    Compute segmentation on the image, return first mask obtained from detectron. Possible to refine the output mask using grabcut. CocoClass default is 0 (person), use 1 for bicycle and 24 for backpack
    '''
    im = cv2.imread(image) if isinstance(image, str) else image
    outputs = predictor(im)
    instances = outputs["instances"]
    instancesOfClass = instances[instances.pred_classes == CocoClass]

    if verbose:
        print("{} Classes found: {}".format(utils.bcolors.presetINFO, outputs["instances"].pred_classes))
        print(f"{utils.bcolors.presetINFO} Number of objects found: {len(instances)}")
        print(f"{utils.bcolors.presetINFO} Number of objects of selected class found: {len(instancesOfClass)}")

    if len(instancesOfClass.pred_masks) > 0:
        mask = instancesOfClass.pred_masks[0]
        mask = np.array(mask.cpu(), dtype=np.uint8)

        mask = Image.fromarray(mask * 255)
        mask.save("./files/temp/mask.jpg")
        mask = cv2.imread("./files/temp/mask.jpg", cv2.IMREAD_GRAYSCALE)

        if (refine):
            print("refining mask...")
            mask = refineMask(mask, im)
            print("done")

        return mask
    else:
        print(f"{utils.bcolors.presetWARNING} No objects of given class found")
        return False


def refineMask(mask, im):
    mask[mask > 0] = cv2.GC_PR_FGD
    mask[mask == 0] = cv2.GC_BGD
    fgModel = np.zeros((1, 65), dtype="float")
    bgModel = np.zeros((1, 65), dtype="float")

    (mask, bgModel, fgModel) = cv2.grabCut(im, mask, None, bgModel, fgModel, iterCount=1, mode=cv2.GC_INIT_WITH_MASK)
    outputMask = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD), 0, 1)
    outputMask = (outputMask * 255).astype("uint8")

    return outputMask

def maskBoundingBox(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = 0, 0, 0, 0
    for c in contours:
        xc, yc, wc, hc = cv2.boundingRect(c)
        if wc > w and hc > h:
            x, y, w, h = xc, yc, wc, hc

    return x, y, w, h


def cutMask(mask, mod ="h", inverse =False, dim=4):
    x, y, w, h = maskBoundingBox(mask)
    height, width= mask.shape

    top = mask.copy()
    # middle = mask.copy()
    bottom = mask.copy()

    # if mod == "h": 
    #     i1 = int(x + w / 3)
    #     i2 = int(i1 + w / 3)
    #     bottom[:, i1: width] = 0
    #     middle[:, 0:i1] = 0
    #     middle[:, i2:width] = 0
    #     top[:, 0:i2] = 0
    # elif mod == "v":
    #     i1 = int(y + h / 3)
    #     i2 = int(i1 + h / 3)
    #     top[i1: height] = 0
    #     middle[0:i1] = 0
    #     middle[i2:height] = 0
    #     bottom[0:i2] = 0
    
    if mod == "v": 
        if inverse:
            i= int(x + w*(dim-1) / dim)
            bottom[:, i:width] = 0
            top[:, 0:i] = 0
        else:
            i = int(x+w / dim)
            bottom[:, 0:i] = 0
            top[:, i:width] = 0
    elif mod == "h":
        i = int(y + h / dim)
        top[i: height] = 0
        bottom[0:i] = 0

    return top, bottom
