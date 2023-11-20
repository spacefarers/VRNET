from dataset_io import Dataset
import model
import os
import train
import fire
import platform

dataset = 'half-cylinder'

if platform.node() == 'PowerPC':
    experiments_dir = '/mnt/d/experiments/'
    root_data_dir = '/mnt/d/data/'
else:
    root_data_dir = '/afs/crc.nd.edu/user/m/myang9/data/'
    experiments_dir = '/afs/crc.nd.edu/user/m/myang9/experiments/'

# root_data_dir = "/Users/spacefarers/data/"
# experiments_dir = "/Users/spacefarers/experiments/"

selected_var = '640'
interval = 2
crop_times = 4
scale = 4
batch_size = 4
load_ensemble_model = True
ensemble_path = experiments_dir + f"ensemble/{dataset}/ensemble.pth"


def run(run_id=102, finetune1_epochs=10, finetune2_epochs=50):
    train_epochs = (finetune1_epochs, finetune2_epochs)
    dataset_io = Dataset(root_data_dir, dataset, selected_var, scale, interval, batch_size)
    dataset_io.load()
    M = model.Net(interval, scale)
    D = model.D()
    model.prep_model(M)
    model.prep_model(D)
    T = train.Trainer(dataset_io, run_id, experiments_dir, M, D)
    if load_ensemble_model:
        T.load_model(ensemble_path)
    # pretrain_time_cost, finetune1_time_cost, finetune2_time_cost, _, _ = T.train(train_epochs[0], train_epochs[1],
    #                                                                              interval, crop_times)
    PSNR, _ = T.inference(interval)
    # T.increase_framerate(interval,"finetune2")
    # return PSNR, pretrain_time_cost, finetune1_time_cost, finetune2_time_cost


if __name__ == "__main__":
    fire.Fire(run)
