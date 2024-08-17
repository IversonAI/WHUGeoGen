"""
Author: Gabe Grand
Improved: WhuDailei
Tools for running inference of a pretrained ControlNet model.
Adapted from gradio_scribble2image.py from the original authors.

"""
import argparse
import sys

sys.path.append('..')
from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dataset import *
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from pytorch_lightning.callbacks import ModelCheckpoint

import time

start_time = time.time()

# 执行你的代码


# import wandb

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--batch_size", type=int, default=16)
# parser.add_argument("--data_dir", type=str, default="/home/user/dailei/Code_4090/ControlNet/training")
parser.add_argument("--data_dir", type=str, default="/data/dailei/WHUGeoGen3")

parser.add_argument("--data_name", type=str, default="BJ_NY")
parser.add_argument("--res_name", type=str, default="data512")
parser.add_argument("--split", type=str, default="finetune")
# parser.add_argument("--load_size", type=int, default=1024)
parser.add_argument("--crop_size", type=int, default=768)
args = parser.parse_args()

data_dir = args.data_dir
data_name = args.data_name
res_name = args.res_name
split = args.split
crop_size = args.crop_size

# Configs
# resume_path = '../models/control_sd15_ini.ckpt'
resume_path = '../models/control_sd21_ini.ckpt'
# resume_path ='/data/dailei/Result/controlnet/models/lr=1e-05_bs=72/lightning_logs/train_NYS16_fine_256_Nearest_crop_14epoch_9h_0409/checkpoints/train_NYS16_fine_256_Nearest_crop_14epoch_9h_0409_epoch=14-step=5999.ckpt'
# resume_path='/home/user/dailei/Code_4090/ControlNet/project/experiments/lr=1e-05_bs=20/lightning_logs/train_NYS16_fine_512_Nearest_crop_9epoch_24h_0410/checkpoints/train_NYS16_fine_512_Nearest_crop_9epoch_24h_0410_epoch=9-step=14399.ckpt'
# resume_path='/home/user/dailei/Code_4090/ControlNet/project/experiments/lr=1e-05_bs=20/lightning_logs/train_NYS16_fine_512_Nearest_crop_unlocked_14epoch_30h_0429/checkpoints/train_NYS16_fine_512_Nearest_crop_unlocked_14epoch_30h_0429.ckpt'
batch_size = args.batch_size
logger_freq = 500
learning_rate = args.lr
sd_locked = True
# sd_locked = False
only_mid_control = False

exp_dir = f'./experiments/lr={learning_rate}_bs={batch_size}_{res_name}_{crop_size}_{data_name}_0.1_{split}_SD2.1'

print("Experiment Directory: ", exp_dir)
print("Learning Rate: ", learning_rate)
print("Batch Size: ", batch_size)
if batch_size >= 5:
    per_device_batch_size =8
    # per_device_batch_size = 72


    grad_acc_steps = int(batch_size / per_device_batch_size)
    # TODO: this won't work for batch sizes that aren't multiples of 4
    assert per_device_batch_size * grad_acc_steps == batch_size
else:
    per_device_batch_size = batch_size
    grad_acc_steps = 1
print("Per Device Batch Size: ", per_device_batch_size)
print("Grad Accumulation Steps: ", grad_acc_steps)

# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
print("Creating models...")
model = create_model('../models/cldm_v21.yaml').cpu()
# model = create_model('../models/cldm_v15.yaml').cpu()

model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control
# Wandb
# wandb_logger = pl.loggers.WandbLogger()
# wandb_logger.experiment.config.update({"batch_size": batch_size, "lr": learning_rate})

# Misc
print("Creating data...")
train_dataset = WhuGeoGenv2Dataset(data_dir=data_dir,  split='finetune', res_name=res_name,data_name=data_name,crop_size=crop_size)
val_dataset = WhuGeoGenv2Dataset(data_dir=data_dir,  split='test', res_name=res_name,data_name=data_name,crop_size=crop_size)
train_dataloader = DataLoader(train_dataset, num_workers=20, batch_size=per_device_batch_size, shuffle=False)
val_dataloader = DataLoader(val_dataset, num_workers=20, batch_size=per_device_batch_size, shuffle=False)
image_logger = ImageLogger(batch_frequency=logger_freq)

# checkpoint_callback = ModelCheckpoint(dirpath='./checkpoints/',save_top_k=-1)
# trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger,checkpoint_callback], max_epochs=21)

checkpoint_callback=pl.callbacks.ModelCheckpoint(save_top_k=-1)
callbacks = [image_logger, checkpoint_callback]

trainer = pl.Trainer(gpus=1, accelerator="gpu", strategy="ddp", precision=32, callbacks=callbacks,
                     default_root_dir=exp_dir, accumulate_grad_batches=grad_acc_steps,limit_train_batches=0.1,limit_val_batches=0.1,max_epochs=6)

# # checkpoint_callback = ModelCheckpoint(dirpath=exp_dir+"/checkpoints/",save_top_k=-1)
#
# stop_callback = pl.callbacks.early_stopping.EarlyStopping(monitor="val/loss", mode="min", patience=7)
# callbacks = [image_logger, stop_callback]
# # callbacks = [stop_callback]
# # trainer = pl.Trainer(gpus=1, precision=32, callbacks=callbacks, default_root_dir=exp_dir, accumulate_grad_batches=grad_acc_steps)
# trainer = pl.Trainer(gpus=2, accelerator="gpu", strategy="ddp", precision=32, callbacks=callbacks,
#                      default_root_dir=exp_dir, accumulate_grad_batches=grad_acc_steps,limit_train_batches=0.1,limit_val_batches=0.1,min_epochs=15, max_epochs=21)



# Train!
trainer.fit(model, train_dataloader, val_dataloader)

end_time = time.time()
execution_time = float((end_time - start_time) / 3600)
print(f"代码执行时间：{execution_time} h")
