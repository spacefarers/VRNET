import platform
import torch
import os
import numpy as np
import neptune
from tqdm import tqdm

machine = platform.node()

# dataset = "half-cylinder"
# target_var = "640"

source_dataset = 'half-cylinder'
source_var = "640"
target_dataset = "vorts"
target_var = "default"

# source_dataset = ["160", "320", "6400"]
# pretrain_vars = ["RAIN", "WSMAG"]

# target_var = "VAPOR"

enable_logging = False
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
    # experiments_dir = '/scratch365/myang9/experiments/'
    # processed_dir = "/scratch365/myang9/processed_data/"
    experiments_dir = '/afs/crc.nd.edu/user/m/myang9/experiments/'
    processed_dir = "/afs/crc.nd.edu/user/m/myang9/data/processed_data"
    # batch_size = 2
    enable_logging = True
elif 'MacBook' in machine or 'mbp' in machine:
    root_data_dir = '/Users/spacefarers/data/'
    experiments_dir = '/Users/spacefarers/experiments/'
    processed_dir = '/Users/spacefarers/data/processed_data/'
elif 'HomePC' in machine:
    root_data_dir = '/mnt/c/Users/spacefarers/data/'
    experiments_dir = '/mnt/c/Users/spacefarers/experiments/'
    processed_dir = '/mnt/c/Users/spacefarers/data/processed_data/'
    # enable_logging = True
    # batch_size = 2
else:
    raise Exception("Unknown machine")

interval = 2
crop_times = 4
# crop_times = 10
# low_res_size for half-cylinder: [160, 60, 20]
# low_res_size for hurricane: [125,125,25]
# low_res_size for vorts: [32,32,32]
crop_size = [16,16,16]  # must be multiples of 8 and smaller than low res size
scale = 4
load_ensemble_model = False
ensemble_path = experiments_dir + f"ensemble/{target_dataset}/ensemble.pth"
run_id = None
tags = [machine, target_dataset]
lr = (1e-5, 4e-5)

pretrain_epochs = 0
finetune1_epochs = 10
finetune2_epochs = 10

train_data_split = 4  # number of datapoints used for training

run_cycle = None
ensemble_iter = None
logging_init = False
enable_restorer = False

print("Machine is", machine)
print(f"Running on {device} with batch size {batch_size}")
print("logging is", "enabled" if enable_logging else "disabled")


def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


seed_everything()
log_obj = None


def init_logging():
    global log_obj
    assert log_obj is None, "run is already set"
    assert run_id is not None, "run_id is not set"
    import keys
    log_obj = neptune.init_run(
        project="VRNET/VRNET",
        api_token=keys.NEPTUNE_API_TOKEN,
        name=f'{run_id:03d}-{machine}',
        tags=tags,
    )
    params = {
        "learning_rate_generator": lr[0],
        "learning_rate_discriminator": lr[1],
        "source_dataset": source_dataset,
        "source_var": source_var,
        "target_dataset": target_dataset,
        "target_var": target_var,
    }
    log_obj["parameters"] = params


domain_backprop = False


def log(data):
    global log_obj
    for key, value in data.items():
        tqdm.write(f"{key}: {value}")
        if enable_logging:
            if log_obj is None:
                init_logging()
            log_obj[key].append(value)


def set_status(status):
    if not enable_logging:
        return
    global log_obj
    if log_obj is None:
        init_logging()
    log_obj["status"] = status
