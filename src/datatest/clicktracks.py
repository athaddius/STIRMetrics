""" clicktracks: allows user to select and view point motion using a tracker and simple opencv visualization
Uses config.json to locate STIR dataset"""

import cv2
import numpy as np
import sys
from tqdm import tqdm
import time
from collections import defaultdict
import itertools

from STIRLoader import STIRLoader
import torch

torch.backends.cudnn.benchmark = True
import argparse
from scipy.spatial import KDTree
from pathlib import Path
import logging
import matplotlib.pyplot as plt
import random
import json
import csrt
from testutil import modeldict
from testutil import todevice
from testutil import showimage


def getargs():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_data",
        type=int,
        default="8",
        help="number of sequences to use",
    )
    parser.add_argument(
        "--modeltype",
        type=str,
        default="CSRT",
        help="CSRT, MFT, RAFT",
    )

    args = parser.parse_args()

    with open("config.json", "r") as f:
        config = json.load(f)
    args.datadir = config["datadir"]
    return args


clicked = False
numpts = 0
POINTSTOTRACK = 4
clickXs = []
clickYs = []


def draw_circle(event, x, y, flags, param):
    global clicked, clickXs, clickYs, numpts
    if event == cv2.EVENT_LBUTTONDOWN:
        clickXs.append(x)
        clickYs.append(y)
        numpts += 1
    if numpts == POINTSTOTRACK:
        clicked = True
        numpts = 0


def trackanddisplay(model, dataloader):
    """tracks and displays pointlist over time
    startpointlist: tensor of [1, numpts, 2] size x,y
    returns pointlist at sequence end"""
    global clicked, clickXs, clickYs

    dataloaderiter = iter(dataloader)
    modelname = str(model.modeltype)
    num_pts = 54
    colors_pts = {
        "CSRTMultiple": np.stack(
            (255 * np.ones(num_pts), 255 * np.ones(num_pts), np.zeros(num_pts)), axis=1
        ),
        "MFT": np.stack(
            (255 * np.ones(num_pts), 255 * np.ones(num_pts), np.zeros(num_pts)), axis=1
        ),
        "RAFT": np.stack(
            (255 * np.ones(num_pts), 255 * np.ones(num_pts), np.zeros(num_pts)), axis=1
        ),
    }

    startdata = todevice(next(iter(dataloader)))
    points_over_time = []
    for i, data in enumerate(tqdm(dataloaderiter)):
        assert len(startdata["ims"]) == 1
        nextdata = todevice(data)
        impair = [
            [*startdata["ims"], *nextdata["ims"]],
            [*startdata["ims_right"], *nextdata["ims_right"]],
        ]
        withcal = True
        if withcal:
            Ks_pair = startdata["Ks"][0]
            Qs_pair = startdata["Qs"][0]
            disparitypad = startdata["disparitypads"][0]
        else:
            Ks_pair = None
            Qs_pair = None
            disparitypad = None
        ims_ori_pair = [*startdata["ims_ori"], *nextdata["ims_ori"]] # RGB image
        imend = ims_ori_pair[1].squeeze(0).numpy()
        imend = cv2.cvtColor(imend, cv2.COLOR_RGB2BGR)
        if i == 0:
            imstart = ims_ori_pair[0].squeeze(0).numpy()
            imstart = cv2.cvtColor(imstart, cv2.COLOR_RGB2BGR)
            while 1:
                showimage("imagetrack", imstart)
                k = cv2.waitKey(20) & 0xFF
                if clicked:
                    break
            startxs = torch.tensor(clickXs)
            startys = torch.tensor(clickYs)
            clicked = False
            clickXs = []
            clickYs = []
            startpointlist = (
                torch.stack((startxs, startys), dim=1)
                .unsqueeze(0)
                .to(device)
                .cpu()
                .squeeze(0)
                .numpy()
            )
            pointlist = (
                startpointlist * 2
            )  # multiply by two since selected on screen which we display with showimage (640x512)

        pointlist = model.trackpoints2D(pointlist, ims_ori_pair)

        VIS = True
        if VIS:
            points_over_time.append(pointlist)
            # loop over all old points [(1,npts,2), (1,npts,2),...]
            points_then_time = np.array(
                [ptlist[:, :].astype(int) for ptlist in points_over_time]
            )
            points_then_time = points_then_time.transpose(1, 0, 2)  # T N 2 -> N T 2

            for ptind, ptlist in enumerate(points_then_time):
                timelen = min(ptlist.shape[0], 48)
                ptlist = ptlist[-timelen:, :]
                curcolor = colors_pts[modelname][ptind]
                increasingdarkness = np.arange(timelen) / timelen
                colors = np.stack(
                    (
                        curcolor[0] * increasingdarkness,
                        curcolor[1] * increasingdarkness,
                        curcolor[2] * increasingdarkness,
                    ),
                    axis=1,
                )
                for pta, ptb, color in zip(ptlist[:-1, :], ptlist[1:, :], colors):
                    pta = tuple(pta)
                    ptb = tuple(ptb)
                    color = color.tolist()
                    imend = cv2.line(imend, pta, ptb, color, thickness=2)

            for pt, color in zip(pointlist[:, :], colors_pts[modelname]):
                pt = pt.astype(int)
                color = color.tolist()
                imend = cv2.circle(imend, tuple(pt), 3, color, thickness=1)
                imend = cv2.circle(imend, tuple(pt), 8, color, thickness=1)

            showimage("imagetrack", imend)
        cv2.waitKey(1)
        startdata = nextdata
    return pointlist, points_over_time


if __name__ == "__main__":
    args = getargs()
    logging.basicConfig(level=logging.INFO)
    device = "cuda"
    modeltype = args.modeltype

    datasets = STIRLoader.getclips(datadir=args.datadir)
    random.shuffle(datasets)
    num_data = args.num_data
    num_data_name = num_data
    if num_data_name == -1:
        num_data_name = "all"
    cv2.namedWindow("imagetrack")
    cv2.setMouseCallback("imagetrack", draw_circle)
    for ind, dataset in enumerate(datasets[:num_data]):
        try:
            model = modeldict[modeltype]() # require re-initialization per-sequence, since CSRT uses templates
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=1, num_workers=0, pin_memory=True
            )
            startseg = np.array(dataset.dataset.getstartseg()).sum(axis=2)
            starticg = dataset.dataset.getstarticg()
            try:
                positionsstart = np.array(dataset.dataset.getstartcenters()) // 2
            except IndexError as e:
                print(f"{e} error on dataset load, continuing")
                continue

            pointlist = torch.from_numpy(positionsstart).unsqueeze(0).to(device)
            print(pointlist.shape)
            endseg = dataset.dataset.getendseg()
            endseg = np.array(endseg).sum(axis=2)
            endicg = dataset.dataset.getendicg()
            centerim = dataset.dataset.getcenters()

            # showimage("seg_start", starticg)
            # showimage("seg_end", endicg)
            # showimage("centerim", centerim)
            _ = dataset.dataset.get3DSegmentationPositions(start=True)  # N 3
            _ = dataset.dataset.get3DSegmentationPositions(start=False)  # N 3

            try:
                positionsend = np.array(dataset.dataset.getendcenters())
            except IndexError as e:
                print(f"{e} error on dataset load, continuing")
                continue

            pointlistend = torch.from_numpy(positionsend).unsqueeze(0).to(device)

            _, points_over_time_2d = trackanddisplay(model, dataloader)
        except AssertionError as e:
            print(f"{e} error on dataset load, continuing")
