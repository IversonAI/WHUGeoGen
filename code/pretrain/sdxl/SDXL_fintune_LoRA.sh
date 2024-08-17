# train LoRA with SDXL

accelerate launch \
  --config_file /home/wyd-7/.cache/huggingface/accelerate/default_config.yaml  \
  --num_cpu_threads_per_process=8 \
  sdxl_train_network.py \
  --sample_prompts="train_config/XL_LoRA_config/sample_prompt.toml" \
  --config_file="train_config/XL_LoRA_config/config_file.toml"

#accelerate launch \
#  --config_file /home/wyd-7/.cache/huggingface/accelerate/default_config.yaml \
#  --num_cpu_threads_per_process=8 \
#  /home/wyd-7/dailei/code/0000_2023_Paper1/02_methods/SDXL-Train/sdxl_train_network.py \
#  --sample_prompts="/home/wyd-7/dailei/code/0000_2023_Paper1/02_methods/SDXL-Train/train_config/XL_LoRA_config/sample_prompt_example.toml" \
#  --config_file="/home/wyd-7/dailei/code/0000_2023_Paper1/02_methods/SDXL-Train/train_config/XL_LoRA_config/config_file_example.toml"


#accelerate launch \
#  --config_file accelerate_config.yaml \
#  --num_cpu_threads_per_process=8 \
#  /home/wyd-7/dailei/code/0000_2023_Paper1/02_methods/SDXL-Train/sdxl_minimal_inference.py \
#  --ckpt_path /home/wyd-7/dailei/code/0000_2023_Paper1/02_methods/SDXL-Train/models/sd_xl_base_1.0.safetensors \
#  --prompt "a satellite image of a city with lots of buildings" \
#  --lora_weights /home/wyd-7/dailei/code/0000_2023_Paper1/02_methods/SDXL-Train/outputs/LoRA/sdxl_lora-000019.safetensors \
#  --output_dir  /home/wyd-7/dailei/code/0000_2023_Paper1/02_methods/SDXL-Train/outputs/LoRA/sample/

#find ./ -name "*.tags" | awk -F "." '{print $2}' | xargs -i -t mv ./{}.tags  ./{}.txt