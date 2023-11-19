import model
from dataset_io import Dataset
import train

dataset = 'half-cylinder'
# root_data_dir = '/mnt/d/data/'
root_data_dir = '/afs/crc.nd.edu/user/m/myang9/data/'
# experiments_dir = '/mnt/d/experiments/'
experiments_dir = '/afs/crc.nd.edu/user/m/myang9/experiments/'

pretrain_vars = ["160", "320", "6400"]

interval = 2
crop_times = 4
scale = 4

if __name__ == "__main__":
    run_id = 101
    M = model.Net(interval, scale)
    D = model.D()
    for ind, var in enumerate(pretrain_vars):
        dataset_io = Dataset(root_data_dir, dataset, var, scale)
        dataset_io.load()
        T = train.Trainer(M, D, dataset_io, run_id, experiments_dir)
        T.train(0, 10, 50, interval, crop_times)
