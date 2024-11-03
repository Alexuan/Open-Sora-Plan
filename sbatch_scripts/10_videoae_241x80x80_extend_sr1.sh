#!/bin/bash

#SBATCH --output=sbatch_outputs/241x80x80_speechlm_bs8_lr2e-5_1img_extend_datacorrect_updatevae.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
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
./scripts/mri_exp/10_videoae_241x80x80_train_extend.sh