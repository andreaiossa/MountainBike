from cv2 import cv2
import numpy as np
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from os.path import isfile
from PIL import Image


def instantiatePredictor():
    '''
    Instantiate detectron2 Mask rcnn predictor from web
    '''
    print("Generating predicor...")
    cfg = get_cfg()
    if (isfile("./config.yaml")):
        cfg.merge_from_file("./config.yaml")
    else:
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

    with open("config.yaml", "w") as f:
        f.write(cfg.dump())

    predictor = DefaultPredictor(cfg)
    print("Done")
    return predictor


def computeSegmentationMask(image, predictor, refine=False):
    '''
    Compute segmentation on the image, return first mask obtained from detectron. Possible to refine the output mask using grabcut
    '''
    print("computing rough mask...")
    im = cv2.imread(image) if isinstance(image, str) else image
    outputs = predictor(im)

    print("classes found: {}".format(outputs["instances"].pred_classes))

    if len(outputs["instances"].pred_masks) > 0:
        mask = outputs["instances"].pred_masks[0]
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
