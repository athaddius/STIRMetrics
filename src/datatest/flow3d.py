"""calculates error of model on STIR labelled dataset.
Averages nearest endpoint error over clips"""

import cv2
import numpy as np
import json
import sys
from tqdm import tqdm
from collections import defaultdict
import itertools

from STIRLoader import STIRLoader
import random
import torch
import argparse
from scipy.spatial import KDTree
from pathlib import Path
import logging
from testutil import *
import raft_stereo_raft
import mft
import MFT.utils.vis_utils as vu
from latency_logger import LatencyLogger, LogLatency

modeldict = {"RAFT_Stereo_RAFT": raft_stereo_raft.RAFTStereoRAFTTracker}


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


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
        default="RAFT_Stereo_RAFT",
        help="Available options: `RAFT_Stereo_RAFT`",
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
    parser.add_argument(
        "--ontestingset",
        type=int,
        default="0",
        help="whether on the testing set. Testing set provides no access to end segs.",
    )
    parser.add_argument(
        "--first_n_frames_to_skip_for_latency_stats",
        type=int,
        default="15",
        help="number of frames to skip for latency stats",
    )
    args = parser.parse_args()
    args.batch_size = 1  # do not change, only one for running
    return args


def drawpoints(im, points, color):
    for pt in points[:, :]:
        pt = pt.astype(int)
        im = cv2.circle(im, tuple(pt), 3, color, thickness=1)
        im = cv2.circle(im, tuple(pt), 12, color, thickness=3)
    return im


def trackanddisplay(
    startpointlist,
    dataloader,
    sequence_name,
    radius=3,
    thickness=-1,
    showvis=False,
    modeltype="CSRT",
    track_writer=None,
):
    """tracks and displays pointlist over time
    returns pointlist at seq end"""
    num_pts = startpointlist.shape[1]
    model = modeldict[modeltype]()
    pointlist = startpointlist
    dataloaderiter = iter(dataloader)
    startdata = todevice(next(dataloaderiter))
    assert len(startdata["ims"]) == 1  # make sure no batches
    colors = (np.random.randint(0, 255, 3 * num_pts)).reshape(num_pts, 3)

    pointlist = pointlist.cpu().numpy()
    firstframe = True
    for data in tqdm(dataloaderiter):
        nextdata = todevice(data)
        ims_ori_pair = [*startdata["ims_ori"], *nextdata["ims_ori"]]
        ims_ori_pair_right = [*startdata["ims_ori_right"], *nextdata["ims_ori_right"]]
        Q_pair = [*startdata["Qs"], *nextdata["Qs"]]
        disparitypads_pair = [*startdata["disparitypads"], *nextdata["disparitypads"]]
        if firstframe and showvis:
            color = [0, 0, 255]
            startframe = drawpoints(
                convert_to_opencv(ims_ori_pair[0]), pointlist, color
            )
            firstframe = False

        with LogLatency(name=sequence_name):
            pointlist, pointlist3d = model.trackpoints3D(
                pointlist, ims_ori_pair, ims_ori_pair_right, Q_pair, disparitypads_pair
            )
        startdata = nextdata
        if showvis:
            imend = convert_to_opencv(ims_ori_pair[1])

            color = [0, 255, 0]
            drawpoints(imend, pointlist, color)

            showimage("imagetrack", imend)
            if track_writer:
                track_writer.write(imend)
            cv2.waitKey(1)
    if showvis:
        lastframe = convert_to_opencv(ims_ori_pair[1])
        return pointlist, pointlist3d, startframe, lastframe
    else:
        return pointlist, pointlist3d


if __name__ == "__main__":
    args = getargs()
    logging.basicConfig(level=logging.INFO)
    modeltype = args.modeltype

    with open("config.json", "r") as f:
        config = json.load(f)
    args.datadir = config["datadir"]
    args.outputdir = config["outputdir"]
    datasets = STIRLoader.getclips(datadir=args.datadir)
    random.seed(1249)
    random.shuffle(datasets)
    errors_avg = defaultdict(int)
    errors_control_avg = 0
    errors_control_avg_3d = 0
    num_data = args.num_data
    num_data_name = num_data
    if num_data_name == -1:
        num_data_name = "all"
        num_data = len(datasets)

    errorlists = {}
    positionlists = {}
    positionlists_3d = {}
    data_used_count = 0
    for ind, dataset in enumerate(datasets[:num_data]):
        try:
            outdir = Path(f"{args.outputdir}/{ind:03d}{modeltype}_tracks.mp4")
            if args.showvis:
                track_writer = vu.VideoWriter(outdir, fps=26, images_export=False)
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=args.batch_size, num_workers=0, pin_memory=True
            )
            startseg = np.array(dataset.dataset.getstartseg()).sum(axis=2)
            try:
                positions_start = np.array(dataset.dataset.getstartcenters())
                if positions_start.shape[0] < 1:
                    continue
                _, _, positions_3d_start = dataset.dataset.get3DSegmentationPositions(
                    True
                )

            except IndexError as e:
                print(f"{e} error on dataset load, continuing")
                continue
            positions_start = torch.from_numpy(positions_start).to(device)
            positions_3d_start = torch.from_numpy(positions_3d_start).to(device)
            if not args.ontestingset:
                endseg = dataset.dataset.getendseg()
                endseg = np.array(endseg).sum(axis=2)
                try:
                    positions_end = np.array(dataset.dataset.getendcenters())
                    _, _, positions_3d_end = dataset.dataset.get3DSegmentationPositions(
                        start=False
                    )

                except IndexError as e:
                    print(f"{e} error on dataset load, continuing")
                    continue
                positions_end = torch.from_numpy(positions_end).to(device)
                positions_3d_end = torch.from_numpy(positions_3d_end).to(device)
                errors_control = pointlossunidirectional(
                    positions_start, positions_end
                )["averagedistance"]
                errors_control_3d = pointlossunidirectional(
                    positions_3d_start, positions_3d_end
                )["averagedistance"]
            if args.showvis:
                showimage("seg_start", startseg)
                showimage("seg_end", endseg)
                cv2.waitKey(1)
                end_estimates, end_estimates_3d, startframe, lastframe = (
                    trackanddisplay(
                        positions_start,
                        dataloader,
                        sequence_name=str(dataset.dataset.basename),
                        showvis=args.showvis,
                        modeltype=modeltype,
                        track_writer=track_writer,
                    )
                )
            else:
                end_estimates, end_estimates_3d = trackanddisplay(
                    positions_start,
                    dataloader,
                    sequence_name=str(dataset.dataset.basename),
                    showvis=args.showvis,
                    modeltype=modeltype,
                )

            positionlists[str(dataset.dataset.basename)] = end_estimates
            positionlists_3d[str(dataset.dataset.basename)] = end_estimates_3d

            if not args.ontestingset:
                errortype = "endpointerror"
                print(f"DATASET_{ind}: {dataset.dataset.basename}")
                print(f"{errortype}_control: {errors_control}")
                print(f"{errortype}_control_3d: {errors_control_3d}")
                errors_control_avg = errors_control_avg + errors_control
                errors_control_avg_3d = errors_control_avg_3d + errors_control_3d
                errordict = {}
                errordict[f"{errortype}_control"] = errors_control
                errordict[f"{errortype}_control_3d"] = errors_control_3d

                errors = pointlossunidirectional(end_estimates, positions_end)
                errors_3d = pointlossunidirectional(end_estimates_3d, positions_3d_end)
                errors_imgavg = errors["averagedistance"]
                errors_imgavg_3d = errors_3d["averagedistance"]
                errorname = f"{errortype}_{modeltype}"
                errordict[errorname] = errors_imgavg
                errordict[errorname + "_3d"] = errors_imgavg_3d
                print(f"{errorname}: {errors_imgavg}")
                print(f"{errorname}_3d: {errors_imgavg_3d}")
                errors_avg[modeltype] = errors_avg[modeltype] + errors_imgavg
                errors_avg[modeltype + "_3d"] = (
                    errors_avg[modeltype + "_3d"] + errors_imgavg_3d
                )
                errorlists[str(dataset.dataset.basename)] = errordict
            data_used_count += 1

            if args.showvis:

                imend = lastframe
                color = [0, 255, 0]
                drawpoints(imend, end_estimates, color)

                displacements = errors["displacements"]
                for pt, displacement in zip(end_estimates, displacements):
                    pt = pt.astype(int)
                    displacement = displacement.astype(int)
                    color = [0, 0, 255]
                    if len(displacement) == 1:
                        print(displacement)
                        continue
                    imend = cv2.line(imend, pt, pt + displacement, color, thickness=2)

                showimage("startframe", startframe)
                showimage("lastframe", imend)
                cv2.waitKey(1)

            if args.showvis:
                track_writer.close()
        except AssertionError as e:
            print(f"error on dataset load, continuing")

    if not args.ontestingset:
        print(f"TOTALS:")
        errors_control_avg = errors_control_avg / data_used_count
        errors_control_avg_3d = errors_control_avg_3d / data_used_count
        print(f"{errortype}_control: {errors_control_avg}")
        print(f"{errortype}_control_3d: {errors_control_avg_3d}")
        errordict = {}
        errordict[f"mean_{errortype}_control"] = errors_control_avg
        errordict[f"mean_{errortype}_control_3d"] = errors_control_avg_3d
        for model, avg in errors_avg.items():
            errorname = f"mean_{errortype}_{model}"
            error = avg / data_used_count
            errordict[errorname] = error
            print(f"{errorname}: {error}")
        errorlists["total"] = errordict
        with open(
            f"{args.outputdir}/{errortype}{num_data_name}{modeltype}{args.jsonsuffix}.json",
            "w",
        ) as fp:
            json.dump(errorlists, fp)
    with open(
        f"{args.outputdir}/positions_{num_data_name}{modeltype}{args.jsonsuffix}.json",
        "w",
    ) as fp:
        json.dump(positionlists, fp, cls=NumpyEncoder)
    with open(
        f"{args.outputdir}/positions3d_{num_data_name}{modeltype}{args.jsonsuffix}.json",
        "w",
    ) as fp:
        json.dump(positionlists_3d, fp, cls=NumpyEncoder)
    LatencyLogger.export_latencies_to_json(
        f"{args.outputdir}/latencies_{num_data_name}{modeltype}{args.jsonsuffix}.json"
    )
    LatencyLogger.export_latency_stats_to_json(
        f"{args.outputdir}/latency_stats_{num_data_name}{modeltype}{args.jsonsuffix}.json",
        first_n_frames_to_skip=args.first_n_frames_to_skip_for_latency_stats,
    )
