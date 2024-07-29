# STIRMetrics

STIRMetrics is an evaluation framework for the [STIR Challenge](https://www.synapse.org/Synapse:syn54126082/wiki/626617) accuracy metrics. This repo provides CSRT, MFT, and RAFT baselines for 2D, and a simple RAFT baseline for 3D tracking. Modify the code as needed to run your tracking algorithm and output point tracks.

## Registration and Prizes

See the [STIR Challenge](https://www.synapse.org/Synapse:syn54126082/wiki/626617) for details.

## Usage

### In your python environment (useful for visualization applications)

Install [STIRLoader](https://github.com/athaddius/STIRLoader) using pip.
If using MFT, install the MFT adapter [MFT_STIR](https://github.com/athaddius/STIRLoader), or if you would like to run models without MFT, comment out all usages of MFT from the codebase.

Set `datadir` in config.json to point to the extracted STIR validation dataset directory.


```
git clone STIRMetrics
cd STIRMetrics/src
```
All commands can be run from the source directory.

### Usage with docker

Follow the docker installation instructions at [STIRHoloscan](https://github.com/athaddius/STIRHoloscan). Afterwards, you can use the rundocker script here:
```
./rundocker.sh <path_to_data>
cd src
```

Now you should be able to run the same commands (with visualization disabled `-showvis 0`).

## Utilities

### clicktracks:
A click-and track application for visualizing a tracker on STIR.

```
python datatest/clicktracks.py --num_data 3
```

### flow2d:

Uses the labelled STIR segments to evaluate tracking performance of given model.

```
python datatest/flow2d.py --num_data 4 --showvis 1
```
Writes output json of averaged results to a json file in the folder results.


### flow3d:

An extension of the flow2d that evaluates the tracking performance in 3D as well.

```
python datatest/flow3d.py --num_data 4 --showvis 1
```
Writes output json of averaged results (3d is in mm) to a json file in the folder results.


## Calculating Error Threshold Metrics for Challenge Evaluation

Firstly, generate ground truth json locations using the following command. Swap `write2d` for `write3d` to generate the 3D positions. Files are written to the `./results` directory.

```
python datatest/write2dgtjson.py --num_data -1 --jsonsuffix test
```

For challenge submissions on the test set, the ground truth location files will be generated, and run independently by the challenge organizers.




Then generate a json file of tracked points predicted by your model using (select whatever you want when testing for `num_data`, and modify the random seed for different sets).

```
python datatest/flow2d.py --num_data 7 --showvis 0 --jsonsuffix test --modeltype <YOURMODEL> --ontestingset 1
```
or for 3d:
```
python datatest/flow3d.py --num_data 7 --showvis 0 --jsonsuffix test --modeltype <YOURMODEL> --ontestingset 1
```

For the challenge, we will run your model on the testing set with `--num_data -1`. Ensure your model executes in a reasonable amount of time. Set your modeltype, or use MFT, RAFT, CSRT to see baseline results for 2D (`RAFT_Stereo_RAFT` is the only type available for 3D). `ontestingset` must be set to 1 for the docker submission, since your model will **not** have access to the ending segmentation locations. For the challenge we will be running your model via flow2d/3d. Thus we recommend not modifying the interface to this file.


The generated ground truth files (start and end gt locations) and your estimates can then be passed into the metric calculator with this:

```
python datatest/calculate_error_from_json2d.py --startgt results/gt_positions_start_all_test.json --endgt results/gt_positions_end_all_test.json  --model_predictions results/positions_<numdata><YOURMODEL>test.json
python datatest/calculate_error_from_json3d.py --startgt results/gt_3d_positions_start_all_test.json --endgt results/gt_3d_positions_end_all_test.json  --model_predictions results/positions3d_<numdata><YOURMODEL>test.json
```

This will print threshold metrics for your model, and a control version of zero-motion. For the challenge, this script will be run by organizers on your `positions.json` file. For the 3D submissions, we will evaluate metrics for both the 2d and 3d locations.
