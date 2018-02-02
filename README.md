## Installation
```
sudo pip install virtualenv           # This may already be installed
virtualenv .env                       # Create a virtual environment
source .env/bin/activate              # Activate the virtual environment
pip install -r requirements.txt       # Install dependencies
```

## Data preparation
Two steps should be performed to prepare data:

+ run the `download_data.sh` to get the train data
+ execute the `tokenizer.sh` to clean up the data for training

## Language detection
Use this docker image `kwakinalabs/langdetect`

## Set up instructions NVIDIA
+ `docker.io` package does not work for `--runtime=nvidia`, one would have to use `Docker-CE`.
+ Installation instructions for [Docker-CE](https://docs.docker.com/install/linux/docker-ce/ubuntu/).
+ Then one would have to install [nvidia-docker-v2](https://github.com/NVIDIA/nvidia-docker/wiki/Installation-(version-2.0))

## Result
| #  | Score   | Description     | Image                    |
|----|---------|-----------------|--------------------------|
| 1  | 0.03778 | `baseline`      | ashmat98/baseline        |
| 2  | 0.02077 | `lang-detect`   | kwakinalabs/langdetect   |
| 3  | 0.02662 | `supervised`    | kwakinalabs/supervised-gpu |
| 4  | 0.00262 | `unmt-v1`       | kwakinalabs/unmt-v1      |

## torch docker image
`kwakinalabs/cuda-torch-cuda-v8`

