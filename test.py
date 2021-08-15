from cv2 import cv2
import numpy as np
from matplotlib import pyplot as plt
import segmentation


def background(video, start, end, tresh, show=False):
    cap = cv2.VideoCapture(video)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(length)
    cap.set(1, start)
    fgbg = cv2.createBackgroundSubtractorMOG2(varThreshold=tresh)
    counter = 0

    img = False
    mask = False

    while (1):
        counter += 1
        print(counter)
        ret, frame = cap.read()
        fgmask = fgbg.apply(frame)

        if show:
            frameR = cv2.resize(frame, (960, 540))
            fgmaskR = cv2.resize(fgmask, (960, 540))
            cv2.imshow('fgmask', fgmaskR)
            cv2.imshow('frame', frameR)

        if counter == end:
            img = frame
            mask = fgmask
            break

        k = cv2.waitKey(30) & 0xff
        if k == 16:
            break

    if show:
        cap.release()
        cv2.destroyAllWindows()

    return img, mask


def remove_noise(mask, radius, iteration1, iteration2):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius, radius))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iteration1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=iteration2)

    return mask


def showImg(images, scale):
    counter = 0
    for image in images:
        counter += 1
        w = int(image.shape[1] / scale)
        h = int(image.shape[0] / scale)
        resized = cv2.resize(image, (w, h))
        cv2.imshow('image {}'.format(counter), resized)
    cv2.waitKey(0)


def compute_hist(reference, img, mask, metric, show=False):
    static_image = cv2.imread(reference)
    predictor = segmentation.create_predictor()
    static_mask = segmentation.compute_mask(reference, predictor, refined=True)
    cut1 = cv2.bitwise_and(static_image, static_image, mask=static_mask)
    cut2 = cv2.bitwise_and(img, img, mask=mask)

    static_image_HSV = cv2.cvtColor(static_image, cv2.COLOR_BGR2HSV)
    img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    channels = [0, 1]
    h_ranges = [0, 180]
    s_ranges = [0, 256]
    ranges = h_ranges + s_ranges

    static_hist = cv2.calcHist([static_image_HSV], channels, static_mask, [8, 8], ranges, accumulate=False)
    static_hist = cv2.normalize(static_hist, static_hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    blob_hist = cv2.calcHist([img_HSV], channels, mask, [8, 8], ranges, accumulate=False)
    blob_hist = cv2.normalize(blob_hist, blob_hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    cut3 = cv2.bitwise_and(static_image_HSV, static_image_HSV, mask=static_mask)
    cut4 = cv2.bitwise_and(img_HSV, img_HSV, mask=mask)

    if show:
        showImg([static_image, static_mask, cut1, cut2, cut3, cut4], 3)

    result = cv2.compareHist(blob_hist, static_hist, metric)
    return result


img, mask = background('./files/videos/single_green_jump.mp4', 1400, 45, 100, show=True)
mask = remove_noise(mask, 5, 1, 1)
result = compute_hist("./files/imgs/blue_front.jpg", img, mask, cv2.HISTCMP_CHISQR, show=False)

print(result)