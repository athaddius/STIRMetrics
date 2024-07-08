""" Writes the start and end segments to a json file."""
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

modeldict = {"MFT": mft.MFTTracker,
           "CSRT": csrt.CSRTMultiple,
           "RAFT": raft.RAFTTracker,}

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

    positionlists_gt_start = {}
    positionlists_gt_end = {}
    for ind, dataset in enumerate(datasets[:num_data]):
        try:
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=args.batch_size, num_workers=0, pin_memory=True
            )
            try:
                pointlist_start = np.array(dataset.dataset.getstartcenters())
                pointlist_end = np.array(dataset.dataset.getendcenters())
            except IndexError as e:
                print(f"{e} error on dataset load, continuing")
                continue
            if pointlist_start.shape[0] < 1:
                continue
            positionlists_gt_start[str(dataset.dataset.basename)] = pointlist_start
            positionlists_gt_end[str(dataset.dataset.basename)] = pointlist_end
        except AssertionError as e:
            print(f"error on dataset load, continuing")

    with open(f'results/gt_positions_start_{num_data_name}_{args.jsonsuffix}.json', 'w') as fp:
        json.dump(positionlists_gt_start, fp, cls=NumpyEncoder)
    with open(f'results/gt_positions_end_{num_data_name}_{args.jsonsuffix}.json', 'w') as fp:
        json.dump(positionlists_gt_end, fp, cls=NumpyEncoder)
