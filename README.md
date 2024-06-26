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
