import platform
import torch

device = platform.node()

dataset = 'half-cylinder'

if device == 'PowerPC':
    experiments_dir = '/mnt/d/experiments/'
    root_data_dir = '/mnt/d/data/'
    processed_dir = '/mnt/d/data/processed_data/'
else:
    root_data_dir = '/afs/crc.nd.edu/user/m/myang9/data/'
    experiments_dir = '/scratch365/myang9/experiments/'
    processed_dir = "/scratch365/myang9/processed_data/"

pretrain_vars = ["160", "320", "6400"]

interval = 2
crop_times = 4
crop_size = [16, 16, 16]
scale = 4
batch_size = 4
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
