from cv2 import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
from datetime import datetime as dt
import time
import datetime
import segmentation
import utils

SAVE_FOLDER = "./files/temp"


def videoInfo(video):
    '''
    Given a video returns encoded vidoe and its basic infos such as: total number of frames, frame rate
    '''
    cap = cv2.VideoCapture(video)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    duration = length / fps if fps > 0 else 0
    duration = f"{int(duration/60)}:{duration%60}"

    return cap, length, fps, duration


def backgroundSub(video, start, end, tresh, show=False):
    '''
    Given a video, the starting frame, the number of frames to parse and the treshold of the BS, it gives the last image and mask for future use.
    '''
    cap, length, fps, duration = videoInfo(video)
    end = length if end > length else end

    print(f"[INFO] number of frames: {length}")
    print(f"[INFO] video duration is {duration} minutes")

    cap.set(1, start)
    fgbg = cv2.createBackgroundSubtractorMOG2(varThreshold=tresh)
    counter = 0

    frame = None
    fgmask = None
    framesAndMasks = []

    while (counter < end):
        counter += 1
        ret, frame = cap.read()
        fgmask = fgbg.apply(frame)
        framesAndMasks.append((frame, fgmask))
        
        if show:
            frameR = cv2.resize(frame, (960, 540))
            fgmaskR = cv2.resize(fgmask, (960, 540))
            cv2.imshow(f'fgmask', fgmaskR)
            cv2.imshow(f'frame', frameR)

            k = cv2.waitKey(30) & 0xff
            if k == 16:
                break

    if show:
        cap.release()
        cv2.destroyAllWindows()

    return framesAndMasks


def bboxBackgroundSub(video, start, end, tresh, show=False, verbose=False, saveMod=False):
    '''
    Given a video, the starting frame, the number of frames to parse and the treshold of the BS, it gives the last image and mask for future use.
    Modality 0 saves frames, modality 1 saves a segment of the video
    '''

    if saveMod:
        now = dt.today().strftime('%Y-%m-%d-%H-%M-%S')
        newPath = SAVE_FOLDER + f"/BS_{now}"
        os.makedirs(newPath)

    cap, length, fps, duration = videoInfo(video)
    end = length if end > length else end

    print(f"[INFO] number of frames: {length}")
    print(f"[INFO] video duration is {duration} minutes")

    cap.set(1, start)
    fgbg = cv2.createBackgroundSubtractorMOG2(varThreshold=tresh, detectShadows=False)
    counter = 0
    riderCounter = 0

    ridersCapture = []
    ridersFrames = []
    box = None
    newRider = True
    previusCut = None
    previusRider = None
    previusMask = None
    previusPercentage = 0
    avgExTime = 0
    exTime = 0
    totExtime = 0
    countTime = 0

    while (counter < end - 1):
        counter += 1
        ret, frame = cap.read()

        if not box:
            cv2.imshow(f'Frame', frame)
            key = cv2.waitKey(30) & 0xff
            if key == ord("s"):
                box = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)

        if all(x is not None for x in [box, frame]):
            startT = time.time()
            if not show:
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
                avgExTime = totExtime / countTime if countTime > 0 else 0
                t = (length * avgExTime) - totExtime
                t = str(datetime.timedelta(seconds=t)).split('.')[0]
                print(f"[INFO] Current frame: {counter+start} over {end}, Riders found: {riderCounter}, Avg Exp Time for frame: {round(avgExTime,2)} sec, Exp time left {t}, Previus bsDiff: {round(previusPercentage,2)}, current bsDiff: {round(percentageDiff,2)}")

            if percentageDiff > 6:
                if percentageDiff > previusPercentage:
                    previusCut = cut
                    previusRider = subFrame
                    previusMask = fgmask
                else:
                    if newRider:
                        riderCounter += 1
                        print(f"[INFO] Rider found with {previusPercentage}")
                        ridersCapture.append((previusRider, previusMask, previusCut, riderCounter))
                        ridersFrames.append((counter + start, riderCounter))
                        newRider = False

                previusPercentage = percentageDiff
            else:
                previusPercentage = 0
                newRider = True

            endT = time.time()
            exTime = endT - startT
            countTime += 1
            totExtime += exTime

    if saveMod == 0:
        for r, m, c, count in ridersCapture:
            cv2.imwrite(os.path.join(newPath, f'rider_{count}.jpg'), r)
            cv2.imwrite(os.path.join(newPath, f'rider_mask_{count}.jpg'), m)
            cv2.imwrite(os.path.join(newPath, f'rider_cut_{count}.jpg'), c)

    if saveMod == 1:
        for f, c in ridersFrames:
            cutVideo(video, f, fps, box, os.path.join(newPath, f'rider_vide_{c}.avi'))

    if saveMod == 2:
        for r, m, c, count in ridersCapture:
            cv2.imwrite(os.path.join(newPath, f'rider_{count}.jpg'), r)
            cv2.imwrite(os.path.join(newPath, f'rider_mask_{count}.jpg'), m)
            cv2.imwrite(os.path.join(newPath, f'rider_cut_{count}.jpg'), c)
        for f, c in ridersFrames:
            print(f"[INFO] Processing video segment of {c} rider")
            cutVideo(video, f, fps, box, os.path.join(newPath, f'rider_vide_{c}.avi'))

    cap.release()
    cv2.destroyAllWindows()

    return ridersCapture, ridersFrames, box


def cutVideo(video, frame, fps, box, path):
    cap = cv2.VideoCapture(video)
    cap.set(1, frame - 70)
    # ret, frame = cap.read()
    # subframe = frame[int(box[1]):int(box[1] + box[3]), int(box[0]):int(box[0] + box[2])]

    frame_width = int(box[2])
    frame_height = int(box[3])

    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (frame_width, frame_height))
    counter = frame - 70
    end = frame + 25
    while (counter <= end):
        counter += 1
        ret, frame = cap.read()
        subframe = frame[int(box[1]):int(box[1] + box[3]), int(box[0]):int(box[0] + box[2])]
        out.write(subframe)
    cap.release()


def detectronOnVideo(video, predictor, refine=False, verbose=False, show=False):
    '''
    Given a video and a detectron predictor it computes a mask for each frame of the video. Returns frame and relative mask in a list
    '''

    cap, length, fps, duration = videoInfo(video)
    print(f"{utils.bcolors.presetINFO} number of frames: {length}")
    print(f"{utils.bcolors.presetINFO} video duration is {duration} minutes")
    frameAndMasks = []
    frameAndMasksFull = []
    counter = 0
    totTime = 0
    while True:
        start = time.time()
        counter += 1
        ret, frame = cap.read()
        if not ret:
            break
        mask = segmentation.computeSegmentationMask(frame, predictor, refine, verbose=False)
        if not isinstance(mask, bool):
            frameAndMasks.append((frame, mask))
        frameAndMasksFull.append((frame, mask))
        end = time.time()
        if verbose:
            currentTime = end - start
            totTime += currentTime
            avgTime = totTime / counter
            expTime = avgTime * (length - counter)
            expTime = str(datetime.timedelta(seconds=expTime)).split('.')[0]
            print(f"{utils.bcolors.presetINFO} Current frame: {counter} over {length}, Avg Exp Time for frame: {round(avgTime,2)} sec, Exp time left {expTime}")

    return frameAndMasks, frameAndMasksFull
