DATASETLOCATION=$1 # Location of the STIR dataset (ie STIROrig or STIRC2024)
OUTPUTLOCATION=$2 # Location to write output to
# Change stirchallenge to your image
TAG=stirchallenge
docker run -it --rm --gpus "device=0" --net host --runtime=nvidia --ipc=host --cap-add=CAP_SYS_PTRACE --ulimit memlock=-1 --ulimit stack=67108864 -v /var/run/docker.sock:/var/run/docker.sock --mount src=$DATASETLOCATION,target=/workspace/data,type=bind --mount src=$OUTPUTLOCATION,target=/workspace/output,type=bind -w /workspace/stir $TAG
