from cv2 import LINE_AA, cv2
import numpy as np
from matplotlib import pyplot as plt

def instantiateYolo():
    '''
    Instantiate yolo network from config file and weights file
    '''

    print("ISTANTIATING YOLO NETWORK...")
    np.random.seed(42)
    classes = open('coco.names').read().strip().split('\n')
    COLORS = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')
    net = cv2.dnn.readNetFromDarknet('yolo/yolov3.cfg', 'yolo/yolov3.weights')
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    print("DONE")

    return classes, COLORS, net, ln


def yoloIdentification(img, h, w, net, ln):
    '''
    Returns full output of yolo nwtwork on given image
    '''
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(ln)

    boxes = []
    confidences = []
    classIDs = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > 0.5:
                box = detection[:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                box = [x, y, int(width), int(height)]
                boxes.append(box)
                confidences.append(float(confidence))
                classIDs.append(classID)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    return boxes, confidences, classIDs, indices


def draw_boxes(img, indices, boxes, COLORS, classIDs, classes, confidences):
    '''
    Given the image which was given to the network it draws the bounding boxes onto it. image must be of same dimension, dont use the image to future to compute mask.
    '''
    boxImg = img
    if len(indices) > 0:
        for i in indices.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(classes[classIDs[i]], confidences[i])
            cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return boxImg


def computeBBoxMask(im, box, refine=False):
    '''
    Given a bounding box in the image (computed using yolo) it returs a probable mask with grabCut.
    '''
    fgModel = np.zeros((1, 65), dtype="float")
    bgModel = np.zeros((1, 65), dtype="float")
    mask = np.zeros(im.shape[:2], dtype="uint8")

    (outputMask, bgModel, fgModel) = cv2.grabCut(im, mask, box, bgModel, fgModel, iterCount=5, mode=cv2.GC_INIT_WITH_RECT)
    if refine:
        outputMask = np.where((outputMask == cv2.GC_BGD) | (outputMask == cv2.GC_PR_BGD), 0, 1)
        outputMask = (outputMask * 255).astype("uint8")

    return outputMask