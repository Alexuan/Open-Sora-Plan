#!/bin/bash

#SBATCH --output=sbatch_outputs/10_make_data_a2v.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --time=2-00:00:00
#SBATCH --account=shrikann_35
#SBATCH --mail-type=BEGIN
#SBATCH --mail-user=xuanshi@usc.edu

source ~/.bashrc
eval "$(conda shell.bash hook)"

conda activate opensora
echo 'which python'
echo 'which nvcc'
nvcc --version

rm -rf /scratch1/xuanshi/REPO/Open-Sora-Plan/dataset_correct
cd /scratch1/xuanshi/REPO/Open-Sora-Plan/scripts/data_preparation
python make_data.py --dataset_dir /project/shrikann_35/xuanshi/DATA/SPAN/span_75speakers --output_dir /scratch1/xuanshi/REPO/Open-Sora-Plan/dataset_correct
