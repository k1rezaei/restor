We use LLaMa-Factory[^1] as it provides efficient implementations for finetuning or continual pretraining of language models.

**Corruption:** We use the following configuration (`.yaml`) for <u>corrupting</u> models. Same configuration is used across all different datasets.

```yaml
### model
model_name_or_path: meta-llama/Meta-Llama-3-8B

### method
stage: pt
do_train: true
finetuning_type: lora
lora_target: all

### ddp
ddp_timeout: 180000000


### dataset
dataset: {dataset_name}
template: llama3
overwrite_cache: true
preprocessing_num_workers: 16

### output
output_dir: {outout_direction}
logging_steps: 10
save_steps: 500
plot_loss: true
overwrite_output_dir: true

### train
per_device_train_batch_size: 2
gradient_accumulation_steps: 2
learning_rate: 5.0e-5
num_train_epochs: 5
lr_scheduler_type: cosine
warmup_ratio: 0.1
fp16: true

### eval
val_size: 0.1
per_device_eval_batch_size: 1
eval_strategy: steps
eval_steps: 500 
```

**LIMA:** After unlearning, the model’s weights may become distorted due to gradient ascent, potentially rendering it unusable. To <u>restore its general utility</u>, we apply a lightweight fine-tuning using LIMA[^2]. This ensures that we can evaluate model’s factual knowledge.

```yaml
### model
model_name_or_path: {path_to_unlearned_model}

### method
stage: sft
do_train: true
finetuning_type: lora
lora_target: all

### ddp
ddp_timeout: 180000000

### dataset
dataset: lima
template: llama3
overwrite_cache: true
preprocessing_num_workers: 16

### output
output_dir: {output_dir}
logging_steps: 10
save_steps: 500
plot_loss: true
overwrite_output_dir: true

### train
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
learning_rate: 2.e-5
num_train_epochs: 2
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true

### eval
val_size: 0.1
per_device_eval_batch_size: 1
eval_strategy: steps
eval_steps: 500    
```

two examples of config files can be found in this directory.

[^1]: <small>https://github.com/hiyouga/LLaMA-Factory/tree/main</small>
[^2]: <small>Zhou, Chunting, et al. "Lima: Less is more for alignment." Advances in Neural Information Processing Systems 36 (2024).</small>