""" calculates error of model on STIR labelled dataset.
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

def getargs():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--startgt",
        type=str,
        default="",
        help="output suffix for json",
    )
    parser.add_argument(
        "--endgt",
        type=str,
        default="",
        help="output suffix for json",
    )
    parser.add_argument(
        "--model_predictions",
        type=str,
        default="",
        help="model predictions json",
    )
    args = parser.parse_args()
    args.batch_size = 1  # do not change, only one for running
    return args







if __name__ == "__main__":
    args = getargs()

    with open("config.json", "r") as f:
        config = json.load(f)
    args.datadir = config["datadir"]
    with open(args.model_predictions, "r") as f:
        model_prediction_dict = json.load(f)
    with open(args.startgt, "r") as f:
        start_gt_dict = json.load(f)
    with open(args.endgt, "r") as f:
        end_gt_dict = json.load(f)
    breakpoint()
    errors_control_avg = 0.0
    errors_avg = defaultdict(int)
    errorlists = {}
    data_used_count = 0
    for filename, pointlist_model in model_prediction_dict.items():
        print(f"Calculating error for: {filename}")
        assert filename in start_gt_dict
        assert filename in end_gt_dict
        pointlist_model = np.array(pointlist_model)
        pointlist_start = np.array(start_gt_dict[filename])
        pointlist_end = np.array(end_gt_dict[filename])
        errors_control = pointlossunidirectional(pointlist_start, pointlist_end)["averagedistance"]
        errortype = "endpointerror"
        errordict = {}
        errordict[f"{errortype}_control"] = errors_control
        print(f"{filename}")
        print(f"{errortype}_control: {errors_control}")
        errors_control_avg = errors_control_avg + errors_control






        errors = pointlossunidirectional(pointlist_model, pointlist_end)
        errors_imgavg = errors["averagedistance"]
        modeltype = "CSRT"
        errorname = f"{errortype}_{modeltype}"
        errordict[errorname] = errors_imgavg
        print(f"{errorname}: {errors_imgavg}")
        errors_avg[modeltype] = errors_avg[modeltype] + errors_imgavg
        errorlists[filename] = errordict
        data_used_count += 1


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
    #with open(f'results/{errortype}{num_data_name}{modeltype}{args.jsonsuffix}.json', 'w') as fp:
        #    json.dump(errorlists, fp)
    #with# open(f'results/positions_{num_data_name}{modeltype}{args.jsonsuffix}.json', 'w') as fp:
        #json.dump(positionlists, fp, cls=NumpyEncoder)
