from cv2 import cv2
import numpy as np
from matplotlib import pyplot as plt


def removeMaskNoise(mask, radius, iteration1, iteration2):
    '''
    apply a kernel of given radius to mask. iteration1 is number of iteration of opening kernel, iteration2 is number of iterations of dilation
    '''

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius, radius))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iteration1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=iteration2)

    return mask


def backgroundSub(video, start, end, tresh, show=False):
    '''
    Given a video, the starting frame, the number of frames to parse and the treshold of the BS, it gives the last image and mask for future use.
    '''

    cap = cv2.VideoCapture(video)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = length / fps
    end = length if end > length else end
    print(f"[INFO] number of frames: {length}")
    print(f"[INFO] video duration is {int(duration/60)}:{duration%60} minutes")
    cap.set(1, start)
    fgbg = cv2.createBackgroundSubtractorMOG2(varThreshold=tresh)
    counter = 0

    frame = None
    fgmask = None

    while (counter < end):
        counter += 1
        ret, frame = cap.read()
        fgmask = fgbg.apply(frame)

        if show:
            frameR = cv2.resize(frame, (960, 540))
            fgmaskR = cv2.resize(fgmask, (960, 540))
            cv2.imshow(f'fgmask: {counter}', fgmaskR)
            cv2.imshow(f'frame: {counter}', frameR)

            k = cv2.waitKey(30) & 0xff
            if k == 16:
                break

    if show:
        cap.release()
        cv2.destroyAllWindows()

    return frame, fgmask


def bboxBackgroundSub(video, start, end, tresh, show=False, verbose=False):
    '''
    Given a video, the starting frame, the number of frames to parse and the treshold of the BS, it gives the last image and mask for future use.
    '''

    cap = cv2.VideoCapture(video)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = length / fps
    end = length if end > length else end
    if verbose:
        print(f"[INFO] number of frames: {length}")
        print(f"[INFO] video duration is {int(duration/60)}:{duration%60} minutes")
    cap.set(1, start)
    fgbg = cv2.createBackgroundSubtractorMOG2(varThreshold=tresh, detectShadows=False)
    counter = 0
    riderCounter = 0

    riders = []
    ridersBS = []
    ridersMasks = []
    box = None
    newRider = True
    previusCut = None
    previusRider = None
    previusPercentage = 0

    while (counter < end - 1):
        counter += 1
        ret, frame = cap.read()

        if not box:
            cv2.imshow(f'Frame', frame)
            key = cv2.waitKey(30) & 0xff
            if key == ord("s"):
                box = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)

        if all(x is not None for x in [box, frame]):
            cv2.destroyAllWindows()
            subFrame = frame[int(box[1]):int(box[1] + box[3]), int(box[0]):int(box[0] + box[2])]
            fgmask = fgbg.apply(subFrame)

            cut = cv2.bitwise_and(subFrame, subFrame, mask=fgmask)

            if show:
                cv2.imshow(f'fgmask', fgmask)
                cv2.imshow(f'Frame', subFrame)
                cv2.imshow(f"cut", cut)

                k = cv2.waitKey(30) & 0xff
                if k == 16:
                    break

            percentageDiff = (np.count_nonzero(cut) * 100) / cut.size

            if verbose:
                print(f"[INFO] Current frame: {counter}, Last frame: {end}, Riders found: {riderCounter}, Previus: {previusPercentage}, now: {percentageDiff}")

            if percentageDiff > 6:
                if percentageDiff > previusPercentage:
                    previusCut = cut
                    previusRider = subFrame
                else:
                    if newRider:
                        riderCounter += 1
                        print(f"[INFO] Rider found with {previusPercentage}")
                        riders.append(previusRider)
                        ridersBS.append(previusCut)
                        ridersMasks.append(fgmask)
                        newRider = False

                previusPercentage = percentageDiff
            else:
                previusPercentage = 0
                newRider = True

    cap.release()
    cv2.destroyAllWindows()

    return riders, ridersBS, ridersMasks