accelerate launch \
  --config_file /home/user/.cache/huggingface/accelerate/default_config.yaml \
  --num_cpu_threads_per_process=8 \
  sdxl_train.py \
  --sample_prompts="train_config/XL_config/sample_prompt_example.toml" \
  --config_file="train_config/XL_config/config_file_example.toml"