""" calculates 3d error of model on STIR labelled dataset.
"""
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




def calculate_accuracy(distances):
    thresholds = [2, 4, 8, 16, 32]
    distances = np.array(distances)
    num_samples = len(distances)
    accuracies = []
    for threshold in thresholds:
        number_below_thresh = np.sum(distances <= threshold)
        accuracy_at_thresh = number_below_thresh / num_samples
        accuracies.append(accuracy_at_thresh)
        print(f"{threshold:2d} mm:\t{accuracy_at_thresh:0.5f}")

    avg_accuracy = np.mean(accuracies)
    print(f"Avg:\t{avg_accuracy:0.5f}")



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
    errors_control_avg = 0.0
    errors_avg = defaultdict(int)
    data_used_count = 0
    distances = []
    control_distances = []
    for filename, pointlist_model in model_prediction_dict.items():
        assert filename in start_gt_dict
        assert filename in end_gt_dict
        pointlist_model = np.array(pointlist_model)
        pointlist_start = np.array(start_gt_dict[filename])
        pointlist_end = np.array(end_gt_dict[filename])

        errors_control = pointlossunidirectional(pointlist_start, pointlist_end)
        control_distancelist = errors_control["distancelist"]
        control_distances.extend(control_distancelist)

        errors = pointlossunidirectional(pointlist_model, pointlist_end)
        distancelist = errors["distancelist"]
        distances.extend(distancelist)
        data_used_count += 1
        print(f"Calculated distances for: {filename}")

    print("CONTROL")
    print("-----")
    print("Accuracy")
    print("-----")
    calculate_accuracy(control_distances)
    print("----------------")
    print("Model")
    print("-----")
    print("Accuracy")
    print("-----")
    calculate_accuracy(distances)
    


