# STIRMetrics

STIRMetrics is an evaluation framework for the [2025 STIR Challenge](https://www.synapse.org/Synapse:syn65877821/wiki/). STIRMetrics calculates accuracy metrics to evaluate point tracking in surgical scenarios. STIRMetrics provides baselines to get started. For 2D, we provide implementations of CSRT, MFT, and RAFT. For 3D, we provide a simple RAFT+RAFT-Stereo method. Modify the code as needed to run your tracking algorithm and output point tracks.

**Note:** To obtain the 2024 challenge code, please checkout the tag `STIRC2024`. The 2025 challenge code is this current branch.

## Registration, Prizes, and Submission

See the [2025 STIR Challenge](https://www.synapse.org/Synapse:syn65877821/wiki/) for details.

## Usage

### In your python environment (useful for visualization applications)

Install [STIRLoader](https://github.com/athaddius/STIRLoader) using pip.
If using MFT, install the MFT adapter [MFT_STIR](https://github.com/athaddius/MFT_STIR), or if you would like to run models without MFT, comment out all usages of MFT from the codebase.

**Configuration:** Edit config.json to give the STIR dataset directory and the output directory.
Set `datadir` to point to the extracted STIR validation dataset directory.
Set `outputdir` to point to a folder in which the output estimates will be written.

Then clone the STIRMetrics code, and change to the `src` directory. All commands are run from here.
```
git clone STIRMetrics
cd STIRMetrics/src
```


### Usage with docker

```
docker build -t stirchallenge .
./rundocker.sh <path_to_data> <path_to_outputdir>
cd STIRMetrics/src
```

Now you should be able to run the same commands (with visualization disabled via passing the `-showvis 0` flag to the python applications). Mount a local volume by modifying `rundocker.sh` if you want to edit/change your code while you build your docker container.

## Utilities

### clicktracks:
A click-and track application for visualizing a tracker on STIR. Example usage to track on three clips:

```
python datatest/clicktracks.py --num_data 3
```

### flow2d:
Uses the labelled STIR segments to evaluate tracking performance of given model.

```
python datatest/flow2d.py --num_data 4 --showvis 1
```
This produces an output json of point locations to the output directory.


### flow3d:

An extension of the flow2d that evaluates the tracking performance in 3D as well.

```
python datatest/flow3d.py --num_data 4 --showvis 1
```
This produces output json files of point locations in 2D and 3D (in mm) to the output directory.


## Calculating Error Threshold Metrics for Challenge Evaluation

1. Export ground-truth files.
2. Run your tracker on the data.
3. Calculate metrics between your outputs and the ground truth.

### 1. Generate Ground Truth JSON

Firstly, generate ground truth json locations from the dataset ([STIROrig](https://ieee-dataport.org/open-access/stir-surgical-tattoos-infrared) or [STIRC2024](https://zenodo.org/records/14803158)) using the following command. Swap `write2d` for `write3d` to generate the 3D positions. Files are written to the `./results` directory.

```
python datatest/write2dgtjson.py --num_data -1 --jsonsuffix test
```

**Note:** For challenge submissions on the test set, the ground truth location files will be generated, and run by the challenge organizers on the non-publicly available 2025 challenge test split.

### 2. Generate JSON for your estimated trackes

Then generate a json file of tracked points predicted by your model using (select whatever you want when testing for `num_data`, and modify the random seed for different sets).

```
python datatest/flow2d.py --num_data 7 --showvis 0 --jsonsuffix test --modeltype <YOURMODEL> --ontestingset 1
```
or for 3d:
```
python datatest/flow3d.py --num_data 7 --showvis 0 --jsonsuffix test --modeltype <YOURMODEL> --ontestingset 1
```

**Note:** For the challenge, we will run your model on the testing set with `--num_data -1`. Ensure your model executes in a reasonable amount of time. Set your modeltype, or use MFT, RAFT, CSRT to see baseline results for 2D (`RAFT_Stereo_RAFT` is the only modeltype baseline available for 3D). `ontestingset` must be set to 1 for the docker submission, since your model will **not** have access to the ending segmentation locations. For the challenge we will be running your model via the flow2d/3d commands. Thus we recommend not modifying the command-line interface to this file.


### 3. Calculate metricss

The generated ground truth files (start and end gt locations) and your estimates can then be passed into the metric calculator with this:

```
python datatest/calculate_error_from_json2d.py --startgt results/gt_positions_start_all_test.json --endgt results/gt_positions_end_all_test.json  --model_predictions results/positions_<numdata><YOURMODEL>test.json
python datatest/calculate_error_from_json3d.py --startgt results/gt_3d_positions_start_all_test.json --endgt results/gt_3d_positions_end_all_test.json  --model_predictions results/positions3d_<numdata><YOURMODEL>test.json
```

This will print threshold metrics for your model, alongside metrics for control version of zero-motion. For the challenge, this script will be run by organizers on your `positions.json` file. For the 3D submissions, we will evaluate metrics for both the 2d and 3d locations.
