CUDA_VISIBLE_DEVICES=1 python scripts/LIS.py --batch_size 8 \
                                             --config configs/stable-diffusion/v1-finetune_WHUGeoGenv2.yaml \
                                             --ckpt /home/user/dailei/Code_4090/FreestyleNet/logs/2024-07-24T21-29-55_exp_WHUGeoGenv2_256/checkpoints/last.ckpt \
                                             --dataset WHUGeoGenv2 \
                                             --outdir outputs/WHUGeoGenv2_LIS_256_t2i \
                                             --txt_file /data/dailei/WHUGeoGen3/test/data512/WHUGeoGenv2_val.txt \
                                             --data_root /data/dailei/WHUGeoGen3/finetune/data512 \
                                             --plms \
                                             --H 256 \
                                             --W 256