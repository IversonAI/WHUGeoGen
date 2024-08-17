#CUDA_VISIBLE_DEVICES=0 python main.py --base configs/stable-diffusion/v1-finetune_FLAIR.yaml \
#                                      -t \
#                                      --actual_resume models/ldm/stable-diffusion/sd-v1-4-full-ema.ckpt \
#                                      -n exp_FLAIR \
#                                      --gpus 0, \
#                                      --data_root /data/dailei/FLAIR \
#                                      --train_txt_file /data/dailei/FLAIR/FLAIR_train.txt \
#                                      --val_txt_file /data/dailei/FLAIR/FLAIR_val.txt
CUDA_VISIBLE_DEVICES=0 python main.py --base configs/stable-diffusion/v1-finetune_FLAIR.yaml \
                                      -t \
                                      --actual_resume logs/2024-01-14T11-38-33_exp_FLAIR/checkpoints/epoch=000013.ckpt \
                                      -n exp_FLAIR \
                                      --gpus 0, \
                                      --data_root /data/dailei/FLAIR \
                                      --train_txt_file /data/dailei/FLAIR/FLAIR_train.txt \
                                      --val_txt_file /data/dailei/FLAIR/FLAIR_val.txt
