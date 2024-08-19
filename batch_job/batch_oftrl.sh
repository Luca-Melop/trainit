#!/bin/bash -l

#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=7.0
#$ -l gpu_memory=24G
#$ -l h_rt=8:00:00   # Specify the hard time limit for the job

### Latest batch script updated on 2024/08/19.

cd /projectnb/aclab/luca/trainit
module load python3/3.10.12 cuda/12.2
source env/bin/activate
python check_env.py

# test sgd optimizer
#python train_jax.py logging.wandb_project=log1 logging.wandb_name=4batch_cheat_oftrl optimizer=oftrl train.use_cheat_hints=True train.accumulate_gradients=True train.accumulation_steps=4 train.use_amp=False optimizer.lr_config.lr=0.0012 #optimizer.beta3=0.9 optimizer.hint_method=3 #optimizer.beta3=0.5  optimizer.hint_method=0
#python train_jax.py logging.wandb_project=log1 logging.wandb_name=8batch_ftrl optimizer=ftrl train.use_cheat_hints=False train.accumulate_gradients=True train.accumulation_steps=8 train.use_amp=False optimizer.lr_config.lr=0.0024 #optimizer.beta3=0.9 optimizer.hint_method=3 #optimizer.beta3=0.5  optimizer.hint_method=0
python train_jax.py logging.wandb_project=log1 logging.wandb_name=cheat_oftrl optimizer=oftrl train.use_cheat_hints=True
#python train_jax.py logging.wandb_project=log1 logging.wandb_name=hint3 optimizer=oftrl optimizer.beta3=0.5  optimizer.hint_method=3 #hint method between 0 and 20 (see optimizer/oftrl.py), beta3 is used for the hint calculations
