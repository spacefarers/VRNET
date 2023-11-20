import model
from dataset_io import Dataset
import train
import time
import platform

dataset = 'half-cylinder'

if platform.node() == 'PowerPC':
    experiments_dir = '/mnt/d/experiments/'
    root_data_dir = '/mnt/d/data/'
else:
    root_data_dir = '/afs/crc.nd.edu/user/m/myang9/data/'
    experiments_dir = '/afs/crc.nd.edu/user/m/myang9/experiments/'

pretrain_vars = ["160", "320", "6400"]

interval = 2
crop_times = 4
scale = 4
batch_size = 4

if __name__ == "__main__":
    run_id = 101
    start_time = time.time()
    datasets = [Dataset(root_data_dir, dataset, var, scale, interval, batch_size) for var in pretrain_vars]
    for dataset_io in datasets:
        dataset_io.load(True)
    M = model.Net(interval, scale)
    D = model.D()
    model.prep_model(M)
    model.prep_model(D)
    for cycle in range(20):
        print(f"Cycle {cycle}:")
        for ind, var in enumerate(pretrain_vars):
            print(f"Training on {var}...")
            T = train.Trainer(datasets[ind], run_id, experiments_dir, M, D)
            _, _, _, M, D = T.train(2, 5, interval, crop_times, True)
    end_time = time.time()
    print(f"Total time cost is {end_time - start_time}")
