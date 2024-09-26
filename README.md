# RL4Seg

## Description

This code base allows you to train a segmentation policy on 2D image data.
Refer to : https://www.arxiv.org/abs/2406.17902

## Install

1. Download the repository:
   ```bash
   # clone project
   git clone --recurse-submodules https://github.com/arnaudjudge/RL4Seg
   cd RL4Seg
   ```
2. Create a virtual environment (Conda is strongly recommended):
   ```bash
   # create conda environment
   conda create -n rl4seg python=3.10
   conda activate rl4seg
   ```
3. Install [PyTorch](https://pytorch.org/get-started/locally/) according to instructions. Grab the one with GPU for faster training:
   ```bash
   # example for linux or Windows
   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
   ```
4. Install the project in editable mode and its dependencies:
   ```bash
   pip install -e .

5. Install vital submodule in editable mode (without dependencies):
    ```bash
    cd vital
    pip install -e . --no-deps
    cd ..
    ```

## Run

To run this code, use the runner.py script for individual runs, 
or the auto_iteration.py script for full RL loops.