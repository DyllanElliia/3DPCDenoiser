# 3D Point Cloud Denoising Tools

This repository contains a learning-based point cloud denoising tool. We provide the usage of the environment setup, denoising script, and error evaluation metric script.

## Cloning the Repository

The repository contains submodules, thus please check it out with

```bash
git clone https://github.com/DyllanElliia/3DPCDenoiser.git --recursive
```

## Installation

The code has been tested in the following environment:

- Python 3.11
- CUDA 12.1.1
- PyTorch 2.4.0

### Install via Conda

Our default, provided install method is based on Conda package and environment management:

```bash
conda env create -f env.yaml
conda activate pcdenoise
```

## Datasets

Download our dataset from [this link](https://github.com/DyllanElliia/3DPCDenoiser?tab=readme-ov-file).

Please extract dataset.zip to data folder.

## Train

```bash
cd scripts
CUDA_VISIBLE_DEVICES=<gpu id list> torchrun <torchrun parameters> train_dist.py
```
Please find tunable parameters in the script.


## Denoise

You can use the following command to denoise point clouds within the dataset:
```bash
cd scripts
python test.py --dataset <dataset name> --resolution <point cloud resolution, e.g. 50000_poisson> --noise <noise level, e.g. 0.01>
```

If you only want to denoise a point cloud, please use the following command:
```bash
python test_single.py --input_xyz <input_xyz_path> --output_xyz <output_xyz_path>
```

## Evaluation

Our evaluation script requires you to download the complete dataset.If you have downloaded and extracted all the test data, you can run it like this:
```bash
python eval.py -m <output_xyz_path> -s <dataset_path>
```