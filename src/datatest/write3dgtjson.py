""" Writes the start and end segment 3d positions to a json file. This is used to prep a file for the calculate_error_from_json3d.py used for the metric calculation."""
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
from pathlib import Path
import logging
from testutil import *

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
        "--num_data",
        type=int,
        default="8",
        help="number of sequences to use",
    )
    args = parser.parse_args()
    args.batch_size = 1  # do not change, only one for running
    return args




if __name__ == "__main__":
    args = getargs()
    logging.basicConfig(level=logging.INFO)

    with open("config.json", "r") as f:
        config = json.load(f)
    args.datadir = config["datadir"]
    datasets = STIRLoader.getclips(datadir=args.datadir)
    random.seed(1249)
    random.shuffle(datasets)
    num_data = args.num_data
    num_data_name = num_data
    if num_data_name == -1:
        num_data_name = "all"

    positionlists_gt_3d_start = {}
    positionlists_gt_3d_end = {}
    for ind, dataset in enumerate(datasets[:num_data]):
        try:
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=args.batch_size, num_workers=0, pin_memory=True
            )
            try:
                positions_s, _, pointlist_3d_start = dataset.dataset.get3DSegmentationPositions(True)
                if positions_s.shape[0] < 1:
                    continue
                _, _, pointlist_3d_end = dataset.dataset.get3DSegmentationPositions(start=False)
            except IndexError as e:
                print(f"'{e}' No segments for matches, continuing. No need to worry")
                continue
            positionlists_gt_3d_start[str(dataset.dataset.basename)] = pointlist_3d_start
            positionlists_gt_3d_end[str(dataset.dataset.basename)] = pointlist_3d_end
        except AssertionError as e:
            print(f"error on dataset load, continuing")

    with open(f'results/gt_3d_positions_start_{num_data_name}_{args.jsonsuffix}.json', 'w') as fp:
        json.dump(positionlists_gt_3d_start, fp, cls=NumpyEncoder)
    with open(f'results/gt_3d_positions_end_{num_data_name}_{args.jsonsuffix}.json', 'w') as fp:
        json.dump(positionlists_gt_3d_end, fp, cls=NumpyEncoder)
