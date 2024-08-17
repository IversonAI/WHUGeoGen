CUDA_VISIBLE_DEVICES=1 python scripts/LIS_single.py --batch_size  1\
                                             --config configs/stable-diffusion/v1-finetune_WHUGeoGenv2.yaml \
                                             --ckpt /home/user/dailei/Code_4090/FreestyleNet/logs/2024-07-26T00-36-36_exp_WHUGeoGenv2_512/checkpoints/last.ckpt \
                                             --dataset WHUGeoGenv2 \
                                             --outdir outputs/WHUGeoGenv2_LIS_single \
                                             --txt_file /data/dailei/WHUGeoGen3/test/data512/WHUGeoGenv2_val.txt \
                                             --data_root /data/dailei/WHUGeoGen3/cig \
                                             --plms \
                                             --H 512 \
                                             --W 512