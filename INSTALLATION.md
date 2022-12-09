# ENVIRONMENT INSTALLATION GUIDE
This installation was tested on:
* OS: Ubuntu 20.04
* Python3.8
* CUDA 11.4
## Step 1. Create environment
```
$ python3 -m venv .venv
$ source .venv/bin/activate
(.venv)$ pip install -U pip
```
## Step 2. Install requirements
### Install non-CUDA dependencies
```
(.venv)$ pip install transformers tensorboard sentencepiece nltk hydra-core tensorboardX rouge-metric
```
### Install PyTorch
```
(.venv)$ pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
```
### Install compare-mt package
```
(.venv)$ git clone https://github.com/neulab/compare-mt.git
(.venv)$ cd compare-mt
(.venv)$ python setup.py install
```