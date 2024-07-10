# STIRMetrics

A metric evaluation framework for STIR. Currently uses CSRT as the implemented model.

## Requirements

STIRLoader, cloned at the same directory level

set datadir in config.json to point at extracted STIR directory
```
git clone STIRLoader
git clone STIRMetrics
cd STIRMetrics
```

## clicktracks:
a click-and track application for visualizing tracker on STIR

```
python datatest/clicktracks.py --num_data 3
```

## flow2d:

Uses the labelled STIR segments to evaluate tracking performance of given model.

```
python datatest/flow2d.py --num_data 4 --showvis 1
```
Writes output json of averaged results to a json file in the folder results.


## flow3d:

An extension of the flow2d that evaluates the tracking performance in 3D as well.

```
python datatest/flow3d.py --num_data 4 --showvis 1
```
Writes output json of averaged results (3d is in mm) to a json file in the folder results.


## Calculating Error Threshold Metrics for Challenge Evaluation

Firstly, generate ground truth json locations using the following command:

```
python datatest/write2dgtjson.py --num_data -1 --jsonsuffix test
```

This files will be generated, and run independently by the organizers for the challenge.




Then generate the points predicted by your model using (select whatever you want to test for `num_data`):

```
python datatest/flow2d.py --num_data 7 --showvis 0 --jsonsuffix test --modeltype <YOURMODEL> --ontestingset 1
```

For the challenge, we will run your model on the testing set with `--num_data -1`. Ensure your model executes in a reasonable amount of time. Set your modeltype, or use MFT, RAFT, CSRT to see baseline results. `ontestingset` must be set to 1 for the docker submission, since your model will **not** have access to the ending segmentation locations. For the challenge we will be running your model via flow2d. Thus we recommend not modifying the interface to this file.


These files (start and end gt locations) can then be passed into the metric calculator with this:

```
python datatest/calculate_error_from_json2d.py --startgt results/gt_positions_start_all_test.json --endgt results/gt_positions_end_all_test.json  --model_predictions results/positions_<numdata><YOURMODEL>test.json
```

This will print threshold metrics for your model, and a control version of zero-motion. For the challenge, this script will be run by organizers on your `positions.json` file.

