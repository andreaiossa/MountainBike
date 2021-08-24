from cv2 import LINE_AA, cv2
import numpy as np
from matplotlib import pyplot as plt
import segmentation


def instantiate_yolo():
    np.random.seed(42)
    classes = open('coco.names').read().strip().split('\n')
    COLORS = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')
    net = cv2.dnn.readNetFromDarknet('./yolo/yolov3.cfg', './yolo/yolov3.weights')
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return classes, COLORS, net, ln


def identification(img, h, w, net, ln):
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
    if len(indices) > 0:
        for i in indices.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(classes[classIDs[i]], confidences[i])
            cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


img = cv2.imread("./files/imgs/green_back.jpg")
(H, W) = img.shape[:2]
H = int(H / 3)
W = int(W / 3)
img = cv2.resize(img, (W, H))

(classes, COLORS, net, ln) = instantiate_yolo()
(boxes, confidences, classIDs, indices) = identification(img, H, W, net, ln)
draw_boxes(img, indices, boxes, COLORS, classIDs, classes, confidences)

cv2.imshow('window', img)
cv2.waitKey(0)
cv2.destroyAllWindows()