#!/bin/bash
#SBATCH --job-name=HugoGAN
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

# Send email when done
#sbatch --mail-user=burto732@gmail.com

# output filter
#sbatch --output=GANHugo.out


#conda init
# source /home/student/${User}/.bashrc

# load modules
#JAX CUDA 11.1 Setup
#~module load cuda/11.1
#activate venv
#~source ~/jax-venv/bin/activate
conda activate conda-torch

# go to data directory
cd /home/student/${USER}/Dev/

#print python version
python --version

# run script
python "P3 GAN CNN.py"