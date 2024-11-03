#!/bin/bash

#SBATCH --output=sbatch_outputs/00_causalvae_finetune_bs2_sr5.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:2
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

cd /scratch1/xuanshi/REPO/Open-Sora-Plan
./scripts/mri_exp/00_causalvae_finetune_sr5.sh
