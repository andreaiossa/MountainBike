from cv2 import cv2
import numpy as np
from matplotlib import pyplot as plt

cap = cv2.VideoCapture('./videos/20210701_160705.mp4')
fgbg = cv2.createBackgroundSubtractorMOG2(varThreshold=100)
counter = 0

while (1):
    counter += 1
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)

    frame = cv2.resize(frame, (960, 540))
    fgmask = cv2.resize(fgmask, (960, 540))

    cv2.imshow('fgmask', fgmask)
    cv2.imshow('frame', frame)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
