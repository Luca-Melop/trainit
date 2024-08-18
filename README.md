# LLM Training Pipeline

This repo extends https://github.com/ZQZCalin/trainit. Instructions are available there.

# New Functionality
- implementation of more optimizers
- cheating version for optimistic optimizers --> OFTRL (POC)
- implementation of various hint methods for optimistic optimizers
- gradient accumulation

# Results

## Benchmark Adam & discounted-FTRL (Adam with O2NC Framework)
```bash
python train_jax.py logging.wandb_project=<project-name> logging.wandb_name=<name> optimizer=ftrl
```   
```bash
python train_jax.py logging.wandb_project=<project-name> logging.wandb_name=<name> #set weight decay to 0 for a fair comparison
```   

https://wandb.ai/optimizedlearning/log1/reports/Benchmark-Adam-discounted-FTRL--Vmlldzo5MDU1NjUw
## Hint Methods OFTRL
```bash
python train_jax.py logging.wandb_project=<project-name> logging.wandb_name=<name> optimizer=oftrl optimizer.beta3=0.5  optimizer.hint_method=0 #hint method between 0 and 20 (see optimizer/oftrl.py), beta3 is used for the hint calculations
```
If you use cheating, hint_method has to be "cheating" (i.e. don't specify any hint method, otherwise it will overwrite the actual cheating hint)
https://wandb.ai/optimizedlearning/log1/reports/Cheating-vs-Hints--Vmlldzo5MDUzODYx
## Cheating POC (with two batches per iteration (2x gradient evaluations))
```bash
python train_jax.py logging.wandb_project=<project-name> logging.wandb_name=cheat_oftrl optimizer=oftrl train.use_cheat_hi
```  
https://wandb.ai/optimizedlearning/log1/reports/Cheating-OFTRL-POC-Adam--Vmlldzo5MDU1NzAz
## Gradient Accumulation 8 Batches
```bash
python train_jax.py logging.wandb_project=<project-name> logging.wandb_name=8batch_cheat_oftrl optimizer=oftrl train.use_cheat_hints=True train.accumulate_gradients=True train.accumulation_steps=8 train.use_amp=False optimizer.lr_config.lr=0.0024
``` 

  
```bash
python train_jax.py logging.wandb_project=<project-name> logging.wandb_name=8batch_ftrl optimizer=ftrl train.use_cheat_hints=False train.accumulate_gradients=True train.accumulation_steps=8 train.use_amp=False optimizer.lr_config.lr=0.0024
```

https://wandb.ai/optimizedlearning/log1/reports/Batch-size-8-Cheating-OFTRL-FTRL--Vmlldzo5MDYxMTY5
## Gradient Accumulation different Batch Size Cheating OFTRL
```bash
python train_jax.py logging.wandb_project=<project-name> logging.wandb_name=4batch_cheat_oftrl optimizer=oftrl train.use_cheat_hints=True train.accumulate_gradients=True train.accumulation_steps=4 train.use_amp=False optimizer.lr_config.lr=0.0012
```   

https://wandb.ai/optimizedlearning/log1/reports/Different-batch-sizes-Cheating-OFTRL--Vmlldzo5MDYxMzc4
