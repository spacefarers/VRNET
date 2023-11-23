import platform
import torch
import wandb
from tqdm import tqdm

machine = platform.node()

dataset = 'half-cylinder'

if machine == 'PowerPC':
    experiments_dir = '/mnt/d/experiments/'
    root_data_dir = '/mnt/d/data/'
    processed_dir = '/mnt/d/data/processed_data/'
elif 'crc' in machine:
    machine = 'CRC'
    root_data_dir = '/afs/crc.nd.edu/user/m/myang9/data/'
    experiments_dir = '/scratch365/myang9/experiments/'
    processed_dir = "/scratch365/myang9/processed_data/"
else:
    raise Exception("Unknown machine")

pretrain_vars = ["160", "320", "6400"]

interval = 2
crop_times = 4
crop_size = [16, 16, 16]
scale = 4
batch_size = 2
run_id = 102
load_ensemble_model = False
ensemble_path = experiments_dir + f"ensemble/{dataset}/ensemble.pth"

pretrain_epochs = 0
finetune1_epochs = 10
finetune2_epochs = 10

train_data_split = 20  # percentage of data used for training

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
wandb_init = False


def init_wandb():
    global wandb_init
    wandb_init = True
    wandb.init(
        project='VRNET',
        name=f'{run_id:03d} ({machine})',
        tags=[machine, dataset]
    )


def log(data):
    if not wandb_init:
        init_wandb()
    wandb.log(data)
