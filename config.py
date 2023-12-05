import platform
import torch
import wandb
import os

os.environ["WANDB_SILENT"] = "true"

machine = platform.node()

# dataset = 'argon'
dataset = 'half-cylinder'

if 'PowerPC' in machine:
    experiments_dir = '/mnt/d/experiments/'
    root_data_dir = '/mnt/d/data/'
    processed_dir = '/mnt/d/data/processed_data/'
elif 'crc' in machine:
    machine = 'CRC'
    root_data_dir = '/afs/crc.nd.edu/user/m/myang9/data/'
    experiments_dir = '/scratch365/myang9/experiments/'
    processed_dir = "/scratch365/myang9/processed_data/"
elif 'MacBook' in machine:
    root_data_dir = '/Users/spacefarers/data/'
    experiments_dir = '/Users/spacefarers/experiments/'
    processed_dir = '/Users/spacefarers/data/processed_data/'
else:
    raise Exception("Unknown machine")

pretrain_vars = ["160", "320", "6400"]

interval = 2
crop_times = 4
# crop_times = 10
crop_size = [16, 16, 16]
scale = 4
load_ensemble_model = False
ensemble_path = experiments_dir + f"ensemble/{dataset}/ensemble.pth"
run_id = None
tags = [machine,dataset]
lr=(1e-5,4e-5)

pretrain_epochs = 0
finetune1_epochs = 10
finetune2_epochs = 10

train_data_split = 20  # percentage of data used for training

run_cycle = None
ensemble_iter = None

if torch.cuda.is_available():
    device = torch.device('cuda')
    batch_size = torch.cuda.device_count()
else:
    device = torch.device('cpu')
    batch_size = 4
wandb_init = False

domain_backprop = True

def init_wandb():
    assert run_id is not None, "run_id is not set"
    global wandb_init
    wandb_init = True
    wandb.init(
        project='VRNET',
        name=f'{run_id:03d} ({machine})',
        tags=tags
    )


def log(data):
    if not wandb_init:
        init_wandb()
    wandb.log(data)
    # pass
