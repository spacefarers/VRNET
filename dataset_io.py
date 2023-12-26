import os
import numpy as np
import json
from skimage.transform import resize
from pathlib import Path
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, ConcatDataset
import config


class Dataset:
    def __init__(self, dataset, selected_var, splice_strategy):
        self.dataset = dataset
        self.selected_var = selected_var
        self.hi_res_data_dir = os.path.join(config.processed_dir, self.dataset, self.selected_var,
                                            'high_res/')
        self.lo_res_data_dir = os.path.join(config.processed_dir, self.dataset, self.selected_var,
                                            'low_res/')
        Path(self.hi_res_data_dir).mkdir(parents=True, exist_ok=True)
        Path(self.lo_res_data_dir).mkdir(parents=True, exist_ok=True)

        self.data_dir = config.root_data_dir + self.dataset + '/'
        self.dataset_json = self.data_dir + 'dataset.json'
        assert os.path.exists(self.dataset_json), "dataset.json does not exist at {}".format(self.dataset_json)
        self.json_data = json.load(open(self.dataset_json))
        self.vars = self.json_data['vars']
        self.dims = self.json_data['dims']
        self.total_samples = self.json_data['total_samples']
        assert self.selected_var in self.vars, "selected_var {} not in vars {}".format(self.selected_var, self.vars)
        self.var_dir = self.data_dir + self.selected_var + '/'

        self.hi_res = []
        self.low_res = []
        train_splice = list(range(0, self.total_samples * config.train_data_split // 100))
        all_splice = list(range(self.total_samples))
        if splice_strategy == "train":
            self.splice_strategy = train_splice
        elif splice_strategy == "all":
            self.splice_strategy = all_splice
        else:
            raise ValueError("Invalid splice strategy")

        self.high_res = [None] * self.total_samples
        self.low_res = [None] * self.total_samples

    def prepare_data(self):
        source_data = []
        for i in tqdm(range(1, self.total_samples + 1), leave=False, desc=f"Reading data {self.dataset}-{self.selected_var}"):
            data = np.fromfile(f"{self.var_dir}{self.dataset}-{self.selected_var}-{i}.raw", dtype='<f')
            source_data.append(data)
        source_data = np.asarray(source_data)
        data_max = np.max(source_data)
        data_min = np.min(source_data)
        source_data = 2 * (source_data - data_min) / (data_max - data_min) - 1
        for i in tqdm(range(self.total_samples), desc=f"Writing to files", leave=False):
            data = source_data[i]
            data = data.reshape(self.dims[2], self.dims[1], self.dims[0]).transpose()
            hi_ = data
            data = resize(data, (self.dims[0] // config.scale, self.dims[1] // config.scale, self.dims[2] // config.scale),
                          order=3)
            lo_ = data
            hi_ = hi_.flatten('F')
            lo_ = lo_.flatten('F')
            hi_.tofile(self.hi_res_data_dir + f'{self.dataset}-{self.selected_var}-{i + 1}.raw', format="<f")
            lo_.tofile(self.lo_res_data_dir + f'{self.dataset}-{self.selected_var}-{i + 1}.raw', format="<f")

    def check_processed_data(self):
        for i in range(1, self.total_samples + 1):
            if not os.path.exists(f'{self.hi_res_data_dir}{self.dataset}-{self.selected_var}-{i}.raw') or not os.path.exists(f'{self.lo_res_data_dir}{self.dataset}-{self.selected_var}-{i}.raw'):
                return False
        return True

    def load(self):
        if self.high_res!=[None]*self.total_samples:
            return
        if not self.check_processed_data():
            self.prepare_data()
        for i in tqdm(self.splice_strategy, leave=False, desc=f"Loading data {self.dataset}-{self.selected_var}"):
            hi_ = np.fromfile(f'{self.hi_res_data_dir}{self.dataset}-{self.selected_var}-{i + 1}.raw', dtype='<f')
            hi_ = hi_.reshape(self.dims[2], self.dims[1], self.dims[0]).transpose()
            lo_ = np.fromfile(f'{self.lo_res_data_dir}{self.dataset}-{self.selected_var}-{i + 1}.raw', dtype='<f')
            lo_ = lo_.reshape(self.dims[2] // config.scale, self.dims[1] // config.scale, self.dims[0] // config.scale).transpose()
            self.high_res[i] = hi_
            self.low_res[i] = lo_

    def get_raw_data(self):
        self.load()
        interval_splice = list(range(0, self.total_samples, config.interval + 1))
        low_res_full = []
        high_res_full = []
        for i in interval_splice:
            low_res_cut = np.take(self.low_res, range(i, i + config.interval + 2), axis=0)
            high_res_cut = np.take(self.high_res, range(i, i + config.interval + 2), axis=0)
            low_res_full.append(low_res_cut)
            high_res_full.append(high_res_cut)
        data = torch.utils.data.TensorDataset(torch.FloatTensor(low_res_full), torch.FloatTensor(high_res_full))
        inference_loader = DataLoader(dataset=data, batch_size=self.batch_size)
        return inference_loader
    def get_augmented_data(self):
        self.load()
        num_windows = len(self.splice_strategy) - config.interval - 1
        low_res_full = np.zeros((config.crop_times * num_windows, config.interval + 2, config.crop_size[0], config.crop_size[1], config.crop_size[2]))
        high_res_full = np.zeros((
                config.crop_times * num_windows, config.interval + 2, config.crop_size[0] * config.scale,
                config.crop_size[1] * config.scale,
                config.crop_size[2] * config.scale))
        idx = 0
        for t in range(num_windows):
            low_res_crop, high_res_crop = self.random_crop_data(
                np.take(self.low_res, self.splice_strategy[t:t + config.interval + 2], axis=0),
                np.take(self.high_res, self.splice_strategy[t:t + config.interval + 2], axis=0))
            for j in range(0, config.crop_times):
                low_res_full[idx] = low_res_crop[j]
                high_res_full[idx] = high_res_crop[j]
                idx += 1
        low_res_full = torch.FloatTensor(low_res_full)
        high_res_full = torch.FloatTensor(high_res_full)
        # domain label repeat self.selected_var num_windows*crop_times times
        domain_label = torch.LongTensor(np.repeat(config.pretrain_vars.index(self.selected_var) if config.domain_backprop and 'ensemble_training' in config.tags else 0, num_windows * config.crop_times))
        data = torch.utils.data.TensorDataset(low_res_full, high_res_full, domain_label)
        train_loader = DataLoader(dataset=data, batch_size=config.batch_size, shuffle=True)
        return train_loader

    def random_crop_data(self, low_res_window, high_res_window):
        low_res_crop = []
        high_res_crop = []
        for _ in range(config.crop_times):
            x = np.random.randint(0, self.dims[0] // config.scale - config.crop_size[0])
            y = np.random.randint(0, self.dims[1] // config.scale - config.crop_size[1])
            z = np.random.randint(0, self.dims[2] // config.scale - config.crop_size[2])

            low_res_crop.append(low_res_window[:, x:x + config.crop_size[0], y:y + config.crop_size[1], z:z + config.crop_size[2]])
            high_res_crop.append(high_res_window[:, x * config.scale:(x + config.crop_size[0]) * config.scale,
                  config.scale * y:(y + config.crop_size[1]) * config.scale,
                  config.scale * z:(z + config.crop_size[2]) * config.scale])
        return low_res_crop, high_res_crop


class MixedDataset:
    def __init__(self, datasets):
        self.datasets = datasets

    def get_data(self, splice_strategy):
        # combine 3 dataloaders
        dataloaders = [dataset.get_data(splice_strategy) for dataset in self.datasets]
        concat_loader = DataLoader(dataset=ConcatDataset([d.dataset for d in dataloaders]), batch_size=config.batch_size, shuffle=True)
        return concat_loader
