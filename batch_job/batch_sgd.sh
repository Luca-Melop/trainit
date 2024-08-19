#!/bin/bash -l

#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=7.0
#$ -l gpu_memory=24G
#$ -l h_rt=8:00:00   # Specify the hard time limit for the job

### Latest batch script updated on 2024/05/14.

cd /projectnb/aclab/luca/trainit
module load python3/3.10.12 cuda/12.2
source env/bin/activate
python check_env.py

# test sgd optimizer
python train_jax.py logging.wandb_project=log1 optimizer=sgd optimizer.lr_config.lr=1e-0
python train_jax.py logging.wandb_project=log1 optimizer=sgd optimizer.lr_config.lr=1e-1
python train_jax.py logging.wandb_project=log1 optimizer=sgd optimizer.lr_config.lr=3e-2
python train_jax.py logging.wandb_project=log1 optimizer=sgd optimizer.lr_config.lr=1e-3
python train_jax.py logging.wandb_project=log1 optimizer=sgd optimizer.lr_config.lr=1e-4

