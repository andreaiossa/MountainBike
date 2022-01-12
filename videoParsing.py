import os
import time
import datetime
import numpy as np
from cv2 import cv2
from datetime import datetime as dt
from matplotlib import pyplot as plt
from components.segmentation import *
from components.utils import *

SAVE_FOLDER = "./files/temp"


def videoInfo(video):
    """ 
    Given a video returns encoded vidoe and its basic infos such as: total number of frames, frame rate
    
    Args:
        video (string): path to the video file to process

    Returns:
        cap: videoCapture object from the opencv library,
        length: length of video in frames,
        fps: frame per second of video,
        duration: duration in minutes
    """
    cap = cv2.VideoCapture(video)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    duration = length / fps if fps > 0 else 0
    duration = f"{int(duration/60)}:{duration%60}"

    return cap, length, fps, duration

def downscaleVideo(video, size, path='./files/temp/test.avi' ):
    """
    Auxiliary function. Rescale a video to given size and save it.

    Args:
        video (string): path to video,
        size (tuple (int,int)): res of output video,
        path (string): output path (default = ./files/temp/test.avi)
    """
    cap, length, fps, duration = videoInfo(video)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('./files/temp/test.avi', fourcc, fps, (400, 400))

    while True:
        ret, frame = cap.read()
        if ret == True:
            b = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            out.write(b)
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

def cutVideo(video, frame, fps, box, cutFrames, path):
    """
    Auxiliary function, not meant to used directly.
    Cuts video counting from given frame starting point, only in ROI.

    Args:
        video (string): path to video
        frame (int): index of frame used to count (middle frame in output)
        fps (int): fps of video necessary for cutting
        box: ROI selected during bbsub
        path (string): output path of video
    """
    cap = cv2.VideoCapture(video)
    cap.set(1, frame - cutFrames[0])
    frame_width = int(box[2])
    frame_height = int(box[3])

    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (frame_width, frame_height))
    counter = frame - cutFrames[0]
    end = frame + cutFrames[1]
    while (counter <= end):
        counter += 1
        ret, frame = cap.read()
        subframe = frame[int(box[1]):int(box[1] + box[3]), int(box[0]):int(box[0] + box[2])]
        out.write(subframe)
    cap.release()

def backgroundSub(video, start, end, tresh, size=False, filterPerc=False, show=False):
    """
    Apply bbsub to given video, output frames and masks obtained. If filterPerc is specified only frames with sufficiently movement are saved 

    Args:
        video (string): path to video
        start (int): starting frame
        end (int): ending frame (capped at legth of video) 
        tresh (float): treshold of backgroun substraction (see cv2.createBackgroundSubtractorMOG2 for reference) 
        size (tuple, optional): size of video if downscale required, not applied if not specified
        filterPerc (int, optional): min percentage of movement to be considered in bbsub, not applied if not specified
        show (bool, optional): showing process (significantly slower)

    Returns:
        frameandmasks: array of tuple of frame and related mask,
        bgFrame: first frame of video, used to obtain feature of background for future use
    """
    if size:
        downscaleVideo(video, size)
        video = "./files/temp/test.avi"
    cap, length, fps, duration = videoInfo(video)
    end = length if end > length else end

    print(f"{bcolors.presetINFO} number of frames: {length}")
    print(f"{bcolors.presetINFO} video duration is {duration} minutes")

    cap.set(1, start)
    fgbg = cv2.createBackgroundSubtractorMOG2(varThreshold=tresh)
    counter = 0

    bgFrame = None
    frame = None
    fgmask = None
    framesAndMasks = []

    while (counter < end):
        counter += 1
        ret, frame = cap.read()
        if counter == 1:
            bgFrame = frame
        fgmask = fgbg.apply(frame)
        fgmask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)[1]
        kernel = np.ones((15, 15), np.uint8)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
        if counter > 1:
            if filterPerc:
                cut = cv2.bitwise_and(frame, frame, mask=fgmask)
                percentageDiff = (np.count_nonzero(cut) * 100) / cut.size
                if percentageDiff > filterPerc:
                    framesAndMasks.append((frame, fgmask))
            else:
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

    return framesAndMasks, bgFrame

def ROIBackgroundSub(video, start, end, tresh, show=False, verbose=False, saveMod=False, percenteDiff=6, cutFrames=(70,25)):
    """
    Apply background substraction to given video in the ROI selected. Number of frames is relative to the maximum point of the movement in the bbsub.

    Args:
        video (string): path to video
        start (int): starting frame
        end (int): ending frame (capped at legth of video) 
        tresh (float): treshold of backgroun substraction (see cv2.createBackgroundSubtractorMOG2 for reference)
        show (bool, optional): showing process (significantly slower)
        percentDiff (int): threshold in bbsub to consider rider passing, tune accordingly to situation 
        verbose (bool, optional): print more useful info
        saveMod (int): 
                - 0: save an image for the rider, mask and cut of apex of rider passage
                - 1: save a video of exactly cutFrames[0] + cutFrames[1] frame of the rider in the ROI selected
                - 2: both options cuncurrently
    Returns:
        [type]: [description]
    """

    if saveMod:
        now = dt.today().strftime('%Y-%m-%d-%H-%M-%S')
        newPath = SAVE_FOLDER + f"/BS_{now}"
        os.makedirs(newPath)

    cap, length, fps, duration = videoInfo(video)
    end = length if end > length else end

    print(f"{bcolors.presetINFO} number of frames: {length}")
    print(f"{bcolors.presetINFO} video duration is {duration} minutes")

    cap.set(1, start)
    fgbg = cv2.createBackgroundSubtractorMOG2(varThreshold=tresh, detectShadows=False)

    exTime = 0
    counter = 0
    avgExTime = 0
    totExtime = 0
    countTime = 0
    riderCounter = 0
    previusPercentage = 0
    ridersFrames = []
    ridersCapture = []
    box = None
    newRider = True
    previusCut = None
    previusRider = None
    previusMask = None

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
            fgmask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)[1]
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
                print(f"{bcolors.presetINFO} Current frame: {counter+start} over {end}, Riders found: {riderCounter}, Avg Exp Time for frame: {round(avgExTime,2)} sec, Exp time left {t}, Previus bsDiff: {round(previusPercentage,2)}, current bsDiff: {round(percentageDiff,2)}")

            if percentageDiff > percenteDiff:
                if percentageDiff > previusPercentage:
                    previusCut = cut
                    previusRider = subFrame
                    previusMask = fgmask
                else:
                    if newRider:
                        riderCounter += 1
                        print(f"{bcolors.presetINFO} Rider found with {previusPercentage}")
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
            cutVideo(video, f, fps, box,cutFrames, os.path.join(newPath, f'rider_vide_{c}.avi'))

    if saveMod == 2:
        for r, m, c, count in ridersCapture:
            cv2.imwrite(os.path.join(newPath, f'rider_{count}.jpg'), r)
            cv2.imwrite(os.path.join(newPath, f'rider_mask_{count}.jpg'), m)
            cv2.imwrite(os.path.join(newPath, f'rider_cut_{count}.jpg'), c)
        for f, c in ridersFrames:
            print(f"{bcolors.presetINFO} Processing video segment of {c} rider")
            cutVideo(video, f, fps, box, os.path.join(newPath, f'rider_vide_{c}.avi'))

    cap.release()
    cv2.destroyAllWindows()

    return ridersCapture, ridersFrames, box


# NOT CORE FUNCTIONS


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
        mask = computeSegmentationMask(frame, predictor, refine, verbose=False)
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


def saveVideoBGSub(rider):
    cap1, length, fpsCustom, duration = videoInfo(rider.customVid)
    cap2, length, fpsBack, duration = videoInfo(rider.backVid)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    outBack = cv2.VideoWriter(rider.folder + '_bgSubBack.avi', fourcc, fpsBack, (400, 400))
    outCustom = cv2.VideoWriter(rider.folder + '_bgSubWoods.avi', fourcc, fpsCustom, (400, 400))

    for frame, mask in rider.frameAndMasksBack:
        cut = cv2.bitwise_and(frame, frame, mask=mask)
        outBack.write(cut)

    for frame, mask in rider.frameAndMasksCustom:
        cut = cv2.bitwise_and(frame, frame, mask=mask)
        outCustom.write(cut)

    cap1.release()
    cap2.release()
    outBack.release()
    outCustom.release()
    cv2.destroyAllWindows()