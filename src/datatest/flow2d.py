""" calculates error of model on STIR labelled dataset.
Averages nearest endpoint error over clips"""
import cv2
import numpy as np
import json
import sys
from tqdm import tqdm
from collections import defaultdict
import itertools

base = "../../"  # FIXME: change to package import
sys.path.append(base + "/STIR_dataloader/")
import STIRLoader
import random
import torch
import argparse
from scipy.spatial import KDTree
from pathlib import Path
import logging
from testutil import *
import csrt
import mft

modeldict = {"MFT": mft.MFTTracker,
           "CSRT": csrt.CSRTMultiple}


def convert_to_opencv(image):
    """ " converts N im tensor to numpy in BGR"""
    image = image.squeeze(0).numpy()
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


def getargs():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jsonsuffix",
        type=str,
        default="",
        help="output suffix for json",
    )
    parser.add_argument(
        "--modeltype",
        type=str,
        default="CSRT",
        help="CSRT or MFT",
    )
    parser.add_argument(
        "--showvis",
        type=int,
        default="1",
        help="whether to show vis",
    )
    parser.add_argument(
        "--num_data",
        type=int,
        default="8",
        help="number of sequences to use",
    )
    args = parser.parse_args()
    args.batch_size = 1  # do not change, only one for running
    return args


def drawpoints(im, points, color):
    for pt in points[:, :]:
        pt = pt.astype(int)
        im = cv2.circle(im, tuple(pt), 3, color, thickness=1)
        im = cv2.circle(im, tuple(pt), 8, color, thickness=1)
    return im


def trackanddisplay(
    startpointlist,
    dataloader,
    radius=3,
    thickness=-1,
    showvis=False,
    modeltype="CSRT"
):
    """tracks and displays pointlist over time
    returns pointlist at seq end"""
    num_pts = startpointlist.shape[1]
    pointlists = {}
    model = modeldict[modeltype]()
    pointlist = startpointlist
    dataloaderiter = iter(dataloader)
    startdata = todevice(next(dataloaderiter))
    assert len(startdata["ims"]) == 1  # make sure no batches
    colors = (np.random.randint(0, 255, 3 * num_pts)).reshape(num_pts, 3)

    pointlist = pointlist.squeeze(0).cpu().numpy()
    firstframe = True
    for data in tqdm(dataloaderiter):
        nextdata = todevice(data)
        impair = [
            [*startdata["ims"], *nextdata["ims"]],
            [*startdata["ims_right"], *nextdata["ims_right"]],
        ]
        ims_ori_pair = [*startdata["ims_ori"], *nextdata["ims_ori"]]
        if firstframe and showvis:
            color = [0, 0, 255]
            startframe = drawpoints(
                convert_to_opencv(ims_ori_pair[0]), pointlist, color
            )
            firstframe = False

        pointlist = model.trackpoints2D(pointlist, ims_ori_pair)
        startdata = nextdata
        if showvis:
            imend = convert_to_opencv(ims_ori_pair[1])

            color = [0, 255, 0]
            drawpoints(imend, pointlist, color)

            showimage("imagetrack", imend)
            cv2.waitKey(1)
    pointlists[str(model.modeltype)] = pointlist
    if showvis:
        lastframe = convert_to_opencv(ims_ori_pair[1])
        pointlists[str(model.modeltype)] = pointlist
        return pointlists, startframe, lastframe
    else:
        return pointlists




if __name__ == "__main__":
    args = getargs()
    logging.basicConfig(level=logging.INFO)
    #modeltype = "CSRT"
    modeltype = args.modeltype
    args.modeltype = modeltype

    with open("config.json", "r") as f:
        config = json.load(f)
    args.datadir = config["datadir"]
    datasets = STIRLoader.getclips(datadir=args.datadir)
    random.seed(1249)
    random.shuffle(datasets)
    errors_avg = defaultdict(int)
    errors_control_avg = 0
    num_data = args.num_data
    num_data_name = num_data
    if num_data_name == -1:
        num_data_name = "all"

    errorlists = {}
    data_used_count = 0  # sometimes skips if too few
    for ind, dataset in enumerate(datasets[:num_data]):
        try:
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=args.batch_size, num_workers=0, pin_memory=True
            )
            num_pts = 128
            startseg = np.array(dataset.dataset.getstartseg()).sum(axis=2)
            # sometimes these throw errors
            start3D = None  # dataset.dataset.getstartsegs3D(start=True) # N 3
            end3D = None  # dataset.dataset.getstartsegs3D(start=False) # N 3
            try:
                positions = np.array(dataset.dataset.getstartcenters())
            except IndexError as e:
                print(f"{e} error on dataset load, continuing")
                continue
            if not args.showvis:
                pointlist = torch.from_numpy(positions).unsqueeze(0).to(device)
                print(pointlist.shape)
            else:
                pointlist = torch.from_numpy(positions).unsqueeze(0).to(device)
                print(pointlist.shape)
            endseg = dataset.dataset.getendseg()
            endseg = np.array(endseg).sum(axis=2)
            try:
                positions = np.array(dataset.dataset.getendcenters())
            except IndexError as e:
                print(f"{e} error on dataset load, continuing")
                continue
            h, w = startseg.shape
            pointlistend = torch.from_numpy(positions).unsqueeze(0).to(device)
            errors_control = pointlossunidirectional(
                pointlist.squeeze(0), pointlistend.squeeze(0)
            )
            errors_control = errors_control["averagedistance"]
            errortype = "endpointerror"

            if args.showvis:
                showimage("seg_start", startseg)
                showimage("seg_end", endseg)
                cv2.waitKey(1)
                endres, startframe, lastframe = trackanddisplay(
                    pointlist,
                    dataloader,
                    showvis=args.showvis,
                    modeltype=modeltype
                )
            else:
                endres = trackanddisplay(
                    pointlist,
                    dataloader,
                    showvis=args.showvis,
                )

            print(f"DATASET_{ind}: {dataset.dataset.basename}")
            print(f"{errortype}_control: {errors_control}")
            errors_control_avg = errors_control_avg + errors_control
            errordict = {}
            errordict[f"{errortype}_control"] = errors_control
            for mtype, result in endres.items():
                # result is a list
                errors = pointlossunidirectional(result, pointlistend.squeeze(0))
                errors_imgavg = errors["averagedistance"]
                errorname = f"{errortype}_{mtype}"
                errordict[errorname] = errors_imgavg
                print(f"{errorname}: {errors_imgavg}")
                errors_avg[modeltype] = errors_avg[modeltype] + errors_imgavg
            errorlists[str(dataset.dataset.basename)] = errordict
            data_used_count += 1

            if args.showvis:

                imend = lastframe
                color = [0, 255, 0]
                drawpoints(imend, result, color)

                displacements = errors["displacements"]
                for pt, displacement in zip(result[:, :], displacements):
                    pt = pt.astype(int)
                    displacement = displacement.astype(int)
                    color = [0, 0, 255]
                    if len(displacement) == 1:
                        print(displacement)
                        continue
                    imend = cv2.line(
                        imend, pt, pt + displacement, color, thickness=2
                    )

                showimage("startframe", startframe)
                showimage("lastframe", imend)
                cv2.waitKey(1)

        except AssertionError as e:
            print(f"error on dataset load, continuing")

    print(f"TOTALS:")
    errors_control_avg = errors_control_avg / data_used_count
    print(f"{errortype}_control: {errors_control_avg}")
    errordict = {}
    errordict[f"mean_{errortype}_control"] = errors_control_avg
    for model, avg in errors_avg.items():
        errorname = f"mean_{errortype}_{model}"
        error = avg / data_used_count
        errordict[errorname] = error
        print(f"{errorname}: {error}")
    errorlists['total'] = errordict
    import json
    with open(f'results/{errortype}{num_data_name}{modeltype}{args.jsonsuffix}.json', 'w') as fp:
        json.dump(errorlists, fp)