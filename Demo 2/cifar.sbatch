#sbatch --time=3:00:00 #3 hours

# Send email when done
#sbatch --mail-user=burto732@gmail.com

# output filter
#sbatch --output=cifar_resnet_1.out


#conda init
source /home/student/${User}/.bashrc

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
python test_resnet_cifar.py