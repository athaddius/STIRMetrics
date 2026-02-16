# STIRMetrics

STIRMetrics is an evaluation framework for the [2025 STIR Challenge](https://www.synapse.org/Synapse:syn65877821/wiki/). STIRMetrics calculates accuracy metrics to evaluate point tracking in surgical scenarios. STIRMetrics provides baselines to get started. For 2D, we provide implementations of 1. CSRT, 2. MFT, and 3. RAFT. For 3D, we provide a simple RAFT+RAFT-Stereo method. Modify the code as needed to run your tracking algorithm and output point tracks.

**Note:** To obtain the 2025 (or 2024) challenge code, please checkout the tag `STIRC2025 (STIRC2024)`. The 2025 challenge code is this current branch.

## Results from prior years

- **2025** A presentation on [2025 STIR Challenge page](https://www.synapse.org/Synapse:syn65877821/wiki/) details the 2025 results. A paper will follow.
- **2024** [2024 STIR Challenge Paper](https://arxiv.org/abs/2503.24306)
- **Original Dataset** [Original dataset paper](https://ieeexplore.ieee.org/document/10458702)

**Dataset Download Links**

- [STIRC2024](https://zenodo.org/records/14803158) (2024 Challenge dataset, no overlap with STIROrig)
- [STIROrig](https://ieee-dataport.org/open-access/stir-surgical-tattoos-infrared) (Original large dataset)

**Citations**

```
@misc{schmidtSTIRC2024,
      title={Point Tracking in Surgery--The 2024 Surgical Tattoos in Infrared (STIR) Challenge}, 
      author={Adam Schmidt and Mert Asim Karaoglu and Soham Sinha and Mingang Jang and Ho-Gun Ha and Kyungmin Jung and Kyeongmo Gu and Ihsan Ullah and Hyunki Lee and Jonáš Šerých and Michal Neoral and Jiří Matas and Rulin Zhou and Wenlong He and An Wang and Hongliang Ren and Bruno Silva and Sandro Queirós and Estêvão Lima and João L. Vilaça and Shunsuke Kikuchi and Atsushi Kouno and Hiroki Matsuzaki and Tongtong Li and Yulu Chen and Ling Li and Xiang Ma and Xiaojian Li and Mona Sheikh Zeinoddin and Xu Wang and Zafer Tandogdu and Greg Shaw and Evangelos Mazomenos and Danail Stoyanov and Yuxin Chen and Zijian Wu and Alexander Ladikos and Simon DiMaio and Septimiu E. Salcudean and Omid Mohareri},
      year={2025},
      eprint={2503.24306},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.24306}, 
}
```

```
@ARTICLE{schmidt2024STIROrigDataset,
  author={Schmidt, Adam and Mohareri, Omid and DiMaio, Simon P. and Salcudean, Septimiu E.},
  journal={IEEE Transactions on Medical Imaging}, 
  title={Surgical Tattoos in Infrared: A Dataset for Quantifying Tissue Tracking and Mapping}, 
  year={2024},
  volume={43},
  number={7},
  pages={2634-2645},
  doi={10.1109/TMI.2024.3372828}}
```


## Registration, Prizes, and Submission

The 2025 STIR challenge is complete, please wait for info on the upcoming 2026 challenge.

## Usage

Here we will provide details on how to use STIRMetrics to evaluate + test your tracking code, either with or without docker.

### In your python environment (useful for visualization applications)

Install [STIRLoader](https://github.com/athaddius/STIRLoader) using pip.
If using MFT, install the MFT adapter [MFT_STIR](https://github.com/athaddius/MFT_STIR), or if you would like to run models without MFT, comment out all usages of MFT from the codebase.

Then clone the STIRMetrics code, and change to the `src` directory. All commands are run from here.
```
git clone STIRMetrics
cd STIRMetrics/src
```

**Configuration:** Edit config.json to provide the STIR dataset directory and the output directory.
Set `datadir` to point to the extracted STIR validation dataset directory.
Set `outputdir` to point to a folder in which the output estimates will be written.

### Usage with docker

First, build the docker container, then use `rundocker.sh` to enter into your container. `rundocker.sh` is a wrapper script around `docker run` which mounts the dataset to `/workspace/data`, and a directory for your tracker outputs to `/workspace/output`.

```
docker build -t stirchallenge .
./rundocker.sh <path_to_data> <path_to_outputdir>
cd STIRMetrics/src
```

Now you should be able to run the same commands (with visualization disabled via passing the `-showvis 0` flag to the python applications). You can mount a local volume by modifying `rundocker.sh` if you want to edit/change your code while you build your docker container. If you mount a local volume, before submission you must make sure to copy all code into your docker image so that it is entirely self-contained and references no files on your system.

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

1. Export ground-truth .json files for testing your model. Run export on any of (STIRC2025, STIRC2024, STIROrig) for testing your model.
2. Run your tracker on the data.
3. Calculate error metrics between your outputs and the ground truth.
4. Repeat steps 2-3 until happy with your model, then submit your method as a docker container.

**Note:** For the challenge itself, the organizers will run step 1,2,3 on the unseen data for the challenge. When developing, we recommend following these steps on a mix of STIRC2025, STIRC2024, and STIROrig to assist designing your methods.

### 1. Generate Ground Truth JSON

Generate ground truth json locations from the dataset ([STIROrig](https://ieee-dataport.org/open-access/stir-surgical-tattoos-infrared) or [STIRC2024](https://zenodo.org/records/14803158)) using the following command. Swap `write2d` for `write3d` to generate the 3D positions. Files are written to the `./results` directory.

```
python datatest/write2dgtjson.py --num_data -1 --jsonsuffix test
```

**Note:** For challenge submissions on the test set, the ground truth location files will be generated, and run by the challenge organizers on the non-publicly available 2026 challenge data.

### 2. Run your tracker: Generate JSON for your estimated tracks

Generate a json file of tracked points predicted by your model. When testing, select whatever number of sequences you want to use for `num_data`, and modify the random seed (in the code) to obtain different sets.

```
python datatest/flow2d.py --num_data 7 --showvis 0 --jsonsuffix test --modeltype <YOURMODEL> --ontestingset 1
```
or for 3d:
```
python datatest/flow3d.py --num_data 7 --showvis 0 --jsonsuffix test --modeltype <YOURMODEL> --ontestingset 1
```

**Note:** For the challenge, we will run your model on the testing set with `--num_data -1`. Ensure your model executes in a reasonable amount of time. Set your modeltype, or use MFT, RAFT, CSRT to see baseline results for 2D (`RAFT_Stereo_RAFT` is the only modeltype baseline available for 3D). `ontestingset` must be set to 1 for the docker submission, since your model will **not** have access to the ending segmentation locations. For the challenge we will be running your model via the flow2d/3d commands. Thus we recommend not modifying the command-line interface to this file.


### 3. Calculate metrics

The generated ground truth files (start and end ground truth locations) and your estimates can then be passed into the metric calculator with this:

```
python datatest/calculate_error_from_json2d.py --startgt results/gt_positions_start_all_test.json --endgt results/gt_positions_end_all_test.json  --model_predictions results/positions_<numdata><YOURMODEL>test.json
python datatest/calculate_error_from_json3d.py --startgt results/gt_3d_positions_start_all_test.json --endgt results/gt_3d_positions_end_all_test.json  --model_predictions results/positions3d_<numdata><YOURMODEL>test.json
```

This will print threshold metrics for your model, alongside metrics for control version of zero-motion. For the challenge, this script will be run by organizers on your `positions.json` file. For the 3D submissions, we will evaluate metrics for both the 2d and 3d locations.


### 4. Submit to challenge

You will send a docker image (created with `docker image save`) to the organizers which runs commands in step 2 and exits. This should be provided along your edited version of `./rundocker.sh`. Ensure your docker script writes to `/workspace/output`, and exits. Organizers will not be running commands within your docker container. Make sure to test your container on another machine, or from another directory/environment, to ensure it container works.
