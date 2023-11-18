from dataset_io import Dataset
import model
import os
import train

dataset = 'half-cylinder'
root_data_dir = '/mnt/d/data/'
# root_data_dir = '/afs/crc.nd.edu/user/m/myang9/data/'
experiments_dir = '/mnt/d/experiments/'
# experiments_dir = '/afs/crc.nd.edu/user/m/myang9/experiments/'

# root_data_dir = "/Users/spacefarers/data/"
# experiments_dir = "/Users/spacefarers/experiments/"

selected_var = '640'
run_id = 4
interval = 2
crop_times = 4
scale = 4

def run(train_epochs):
    dataset_io = Dataset(root_data_dir, dataset, selected_var,scale)
    dataset_io.load()
    M = model.Net(interval, scale)
    D = model.D()
    T = train.Trainer(M, D, dataset_io, run_id, experiments_dir)
    T.train(train_epochs[0],train_epochs[1],train_epochs[2], interval, crop_times)
    T.increase_framerate(interval)

if __name__ == "__main__":
    run((0,10,50))