DATASETLOCATION=$1 # Location of the STIR dataset (ie STIROrig or STIRC2024)
OUTPUTLOCATION=$2 # Location to write output to
# Change stirchallenge to your image
TAG=stirchallenge_latency

## For developing your code
#./rundocker.sh /home/adam/data/test/2025/STIRTest_2025 /home/adam/STIR/STIRMetrics_Challenge/results_2025
#docker run -it --rm --gpus "device=0" --net host --runtime=nvidia --ipc=host --cap-add=CAP_SYS_PTRACE --ulimit memlock=-1 --ulimit stack=67108864 -v /var/run/docker.sock:/var/run/docker.sock --mount src=$DATASETLOCATION,target=/workspace/data,type=bind --mount src=$OUTPUTLOCATION,target=/workspace/output,type=bind $TAG
#docker run -it --gpus "device=0" --net host --runtime=nvidia --ipc=host --cap-add=CAP_SYS_PTRACE --ulimit memlock=-1 --ulimit stack=67108864 -v /var/run/docker.sock:/var/run/docker.sock --mount src=$DATASETLOCATION,target=/workspace/data,type=bind --mount src=$OUTPUTLOCATION,target=/workspace/output,type=bind $TAG #/bin/bash -c "cd STIRMetrics/src && python datatest/write2dgtjson.py --num_data -1 --jsonsuffix test && python datatest/write3dgtjson.py --num_data -1 --jsonsuffix test && /bin/bash"
#exit


DEVICE=1
## FOR SUBMISSION -- 2D
## Uncomment the following line to run your model (and comment the above line). Make sure to change modeltype to your model. Modify commands as needed to run your code.
docker run --rm --gpus "device=$DEVICE" --net host --runtime=nvidia --ipc=host --cap-add=CAP_SYS_PTRACE --ulimit memlock=-1 --ulimit stack=67108864 -v /var/run/docker.sock:/var/run/docker.sock --mount src=$DATASETLOCATION,target=/workspace/data,type=bind --mount src=$OUTPUTLOCATION,target=/workspace/output,type=bind $TAG /bin/bash -c "cd STIRMetrics/src && python datatest/flow2d.py --num_data -1 --showvis 0 --jsonsuffix test --modeltype RAFT --ontestingset 1"
docker run --rm --gpus "device=$DEVICE" --net host --runtime=nvidia --ipc=host --cap-add=CAP_SYS_PTRACE --ulimit memlock=-1 --ulimit stack=67108864 -v /var/run/docker.sock:/var/run/docker.sock --mount src=$DATASETLOCATION,target=/workspace/data,type=bind --mount src=$OUTPUTLOCATION,target=/workspace/output,type=bind $TAG /bin/bash -c "cd STIRMetrics/src && python datatest/flow2d.py --num_data -1 --showvis 0 --jsonsuffix test --modeltype CSRT --ontestingset 1"
docker run --rm --gpus "device=$DEVICE" --net host --runtime=nvidia --ipc=host --cap-add=CAP_SYS_PTRACE --ulimit memlock=-1 --ulimit stack=67108864 -v /var/run/docker.sock:/var/run/docker.sock --mount src=$DATASETLOCATION,target=/workspace/data,type=bind --mount src=$OUTPUTLOCATION,target=/workspace/output,type=bind $TAG /bin/bash -c "cd STIRMetrics/src && python datatest/flow2d.py --num_data -1 --showvis 0 --jsonsuffix test --modeltype MFT --ontestingset 1"

## For Submission -- 3D
docker run --rm --gpus "device=$DEVICE" --net host --runtime=nvidia --ipc=host --cap-add=CAP_SYS_PTRACE --ulimit memlock=-1 --ulimit stack=67108864 -v /var/run/docker.sock:/var/run/docker.sock --mount src=$DATASETLOCATION,target=/workspace/data,type=bind --mount src=$OUTPUTLOCATION,target=/workspace/output,type=bind $TAG /bin/bash -c "cd STIRMetrics/src && python datatest/flow3d.py --num_data -1 --showvis 0 --jsonsuffix test --modeltype RAFT_Stereo_RAFT --ontestingset 1"
