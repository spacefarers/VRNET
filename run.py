from dataset_io import Dataset
import model
import os
import train
import fire

dataset = 'half-cylinder'
root_data_dir = '/mnt/d/data/'
# root_data_dir = '/afs/crc.nd.edu/user/m/myang9/data/'
experiments_dir = '/mnt/d/experiments/'
# experiments_dir = '/afs/crc.nd.edu/user/m/myang9/experiments/'

# root_data_dir = "/Users/spacefarers/data/"
# experiments_dir = "/Users/spacefarers/experiments/"

selected_var = '640'
interval = 2
crop_times = 4
scale = 4
batch_size = 4


def run(run_id=1, pretrain_epochs=0, finetune1_epochs=10, finetune2_epochs=50):
    train_epochs = (pretrain_epochs, finetune1_epochs, finetune2_epochs)
    dataset_io = Dataset(root_data_dir, dataset, selected_var, scale, batch_size)
    dataset_io.load()
    M = model.Net(interval, scale)
    D = model.D()
    T = train.Trainer(dataset_io, run_id, experiments_dir,M,D)
    pretrain_time_cost, finetune1_time_cost, finetune2_time_cost = T.train(train_epochs[0], train_epochs[1],
                                                                           train_epochs[2], interval, crop_times)
    PSNR, _ = T.inference(interval,"finetune1")
    # T.increase_framerate(interval)
    return PSNR, pretrain_time_cost, finetune1_time_cost, finetune2_time_cost


if __name__ == "__main__":
    fire.Fire(run)
