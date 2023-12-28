import platform
import torch
import wandb
import os
import numpy as np


os.environ["WANDB_SILENT"] = "true"

machine = platform.node()


dataset = "hurricane"
target_var = "RAIN"

# dataset = 'half-cylinder'
# source_dataset = 'hurricane'

# source_dataset = ["160", "320", "6400"]
# pretrain_vars = ["RAIN", "WSMAG"]

# source_var = "RAIN"
# target_var = "640"
# target_var = "VAPOR"

use_wandb = False
if torch.cuda.is_available():
    device = torch.device('cuda')
    batch_size = torch.cuda.device_count()
else:
    device = torch.device('cpu')
    batch_size = 4

if 'PowerPC' in machine:
    experiments_dir = '/mnt/d/experiments/'
    root_data_dir = '/mnt/d/data/'
    processed_dir = '/mnt/d/data/processed_data/'
elif 'crc' in machine:
    machine = 'CRC'
    root_data_dir = '/afs/crc.nd.edu/user/m/myang9/data/'
    experiments_dir = '/scratch365/myang9/experiments/'
    processed_dir = "/scratch365/myang9/processed_data/"
    batch_size = 2
    use_wandb = True
elif 'MacBook' in machine or 'mbp' in machine:
    root_data_dir = '/Users/spacefarers/data/'
    experiments_dir = '/Users/spacefarers/experiments/'
    processed_dir = '/Users/spacefarers/data/processed_data/'
elif 'HomePC' in machine:
    root_data_dir = '/mnt/c/Users/spacefarers/data/'
    experiments_dir = '/mnt/c/Users/spacefarers/experiments/'
    processed_dir = '/mnt/c/Users/spacefarers/data/processed_data/'
else:
    raise Exception("Unknown machine")

interval = 2
crop_times = 4
# crop_times = 10
# low_res_size for half-cylinder: [160, 60, 20]
# low_res_size for Hurricane: [125,125,25]
crop_size = [16, 16, 16] # must be multiples of 8 and smaller than low res size
scale = 4
load_ensemble_model = False
ensemble_path = experiments_dir + f"ensemble/{dataset}/ensemble.pth"
run_id = None
tags = [machine,dataset]
lr=(1e-4,4e-4)

pretrain_epochs = 0
finetune1_epochs = 10
finetune2_epochs = 10

train_data_split = 20  # percentage of data used for training

run_cycle = None
ensemble_iter = None
wandb_init = False

print("Machine is", machine)
print(f"Running on {device} with batch size {batch_size}")
print("Wandb is", "enabled" if use_wandb else "disabled")


def seed_everything(seed=42):
   np.random.seed(seed)
   torch.manual_seed(seed)
   torch.cuda.manual_seed(seed)
   torch.cuda.manual_seed_all(seed)
   torch.backends.cudnn.benchmark = False
   torch.backends.cudnn.deterministic = True

seed_everything()

def init_wandb():
    assert run_id is not None, "run_id is not set"
    global wandb_init
    wandb_init = True
    wandb.init(
        project='VRNET',
        name=f'{run_id:03d} ({machine})',
        tags=tags
    )

domain_backprop = False

def log(data):
    if not use_wandb:
        return
    if not wandb_init:
        init_wandb()
    wandb.log(data)
