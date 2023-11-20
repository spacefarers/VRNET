import os
import numpy as np
import json
from skimage.transform import resize
from pathlib import Path
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader


class Dataset:
    def __init__(self, root_data_dir, dataset, selected_var, scale, interval, batch_size):
        self.dataset = dataset
        self.root_data_dir = root_data_dir
        self.selected_var = selected_var
        self.batch_size = batch_size
        self.interval = interval
        self.hi_res_data_dir = os.path.join(self.root_data_dir, 'processed_data', self.dataset, self.selected_var,
                                            'high_res/')
        self.lo_res_data_dir = os.path.join(self.root_data_dir, 'processed_data', self.dataset, self.selected_var,
                                            'low_res/')
        Path(self.hi_res_data_dir).mkdir(parents=True, exist_ok=True)
        Path(self.lo_res_data_dir).mkdir(parents=True, exist_ok=True)

        self.data_dir = self.root_data_dir + self.dataset + '/'
        self.dataset_json = self.data_dir + 'dataset.json'
        assert os.path.exists(self.dataset_json), "dataset.json does not exist at {}".format(self.dataset_json)
        self.json_data = json.load(open(self.dataset_json))
        self.vars = self.json_data['vars']
        self.dims = self.json_data['dims']
        self.total_samples = self.json_data['total_samples']
        assert self.selected_var in self.vars, "selected_var {} not in vars {}".format(self.selected_var, self.vars)
        self.var_dir = self.data_dir + self.selected_var + '/'
        self.scale = scale
        self.crop_size = [16, 16, 16]

        self.hi_res = []
        self.low_res = []
        self.hi_res_train = []
        self.low_res_train = []

    def prepare_data(self):
        print("Processing Data...")
        source_data = []
        for i in tqdm(range(1, self.total_samples + 1)):
            data = np.fromfile(f"{self.var_dir}{self.dataset}-{self.selected_var}-{i}.raw", dtype='<f')
            source_data.append(data)
        source_data = np.asarray(source_data)
        data_max = np.max(source_data)
        data_min = np.min(source_data)
        source_data = 2 * (source_data - data_min) / (data_max - data_min) - 1
        for i in tqdm(range(self.total_samples)):
            data = source_data[i]
            data = data.reshape(self.dims[2], self.dims[1], self.dims[0]).transpose()
            hi_ = data
            data = resize(data, (self.dims[0] // self.scale, self.dims[1] // self.scale, self.dims[2] // self.scale),
                          order=3)
            lo_ = data
            hi_ = hi_.flatten('F')
            lo_ = lo_.flatten('F')
            hi_.tofile(self.hi_res_data_dir + f'{self.dataset}-{self.selected_var}-{i + 1}.raw', format="<f")
            lo_.tofile(self.lo_res_data_dir + f'{self.dataset}-{self.selected_var}-{i + 1}.raw', format="<f")

    def check_processed_data(self):
        for i in range(1, self.total_samples + 1):
            if not os.path.exists(
                    f'{self.hi_res_data_dir}{self.dataset}-{self.selected_var}-{i}.raw') or not os.path.exists(
                f'{self.lo_res_data_dir}{self.dataset}-{self.selected_var}-{i}.raw'):
                return False
        return True

    def load(self, use_all_data=False):
        self.hi_res = []
        self.low_res = []
        if not self.check_processed_data():
            self.prepare_data()
        print("Loading Data...")
        for i in tqdm(range(1, self.total_samples + 1)):
            hi_ = np.fromfile(f'{self.hi_res_data_dir}{self.dataset}-{self.selected_var}-{i}.raw', dtype='<f')
            hi_ = hi_.reshape(self.dims[2], self.dims[1], self.dims[0]).transpose()
            lo_ = np.fromfile(f'{self.lo_res_data_dir}{self.dataset}-{self.selected_var}-{i}.raw', dtype='<f')
            lo_ = lo_.reshape(self.dims[2] // self.scale, self.dims[1] // self.scale,
                              self.dims[0] // self.scale).transpose()
            self.hi_res.append(hi_)
            self.low_res.append(lo_)
        self.hi_res = np.asarray(self.hi_res)
        self.low_res = np.asarray(self.low_res)
        self.hi_res_train = np.array(
            [self.hi_res[i] for i in (range(0, self.total_samples, self.interval + 1) if not use_all_data else range(
                self.total_samples))])
        self.low_res_train = np.array(
            [self.low_res[i] for i in (range(0, self.total_samples, self.interval + 1) if not use_all_data else range(
                self.total_samples))])

    def spacial_temporal_dataloader(self, crop_times):
        num_windows = len(self.hi_res_train) - self.interval - 1
        ls_train = np.zeros((crop_times * num_windows, 1, self.crop_size[0], self.crop_size[1], self.crop_size[2]))
        le_train = np.zeros((crop_times * num_windows, 1, self.crop_size[0], self.crop_size[1], self.crop_size[2]))
        hi_train = np.zeros(
            (crop_times * num_windows, self.interval + 2, self.crop_size[0] * self.scale, self.crop_size[1] * self.scale,
             self.crop_size[2] * self.scale))
        idx = 0
        for t in range(num_windows):
            ls_, le_, hi_ = self.spacial_temporal_crop(self.low_res_train[t:t + self.interval + 2],
                                                       self.hi_res_train[t:t + self.interval + 2], crop_times, self.interval)
            for j in range(0, crop_times):
                ls_train[idx] = ls_[j]
                le_train[idx] = le_[j]
                hi_train[idx] = hi_[j]
                idx += 1
        ls_train = torch.FloatTensor(ls_train)
        le_train = torch.FloatTensor(le_train)
        hi_train = torch.FloatTensor(hi_train)
        data = torch.utils.data.TensorDataset(ls_train, le_train, hi_train)
        train_loader = DataLoader(dataset=data, batch_size=self.batch_size, shuffle=True)
        return train_loader

    def spacial_temporal_crop(self, low_res_window, high_res_window, crop_times, interval):
        hi = []
        ls = []
        le = []
        for _ in range(crop_times):
            x = np.random.randint(0, self.dims[0] // self.scale - self.crop_size[0])
            y = np.random.randint(0, self.dims[1] // self.scale - self.crop_size[1])
            z = np.random.randint(0, self.dims[2] // self.scale - self.crop_size[2])

            ls_ = low_res_window[0:1, x:x + self.crop_size[0], y:y + self.crop_size[1], z:z + self.crop_size[2]]
            le_ = low_res_window[interval + 1:interval + 2, x:x + self.crop_size[0], y:y + self.crop_size[1],
                  z:z + self.crop_size[2]]
            hi_ = high_res_window[:, x * self.scale:(x + self.crop_size[0]) * self.scale,
                  self.scale * y:(y + self.crop_size[1]) * self.scale,
                  self.scale * z:(z + self.crop_size[2]) * self.scale]
            ls.append(ls_)
            hi.append(hi_)
            le.append(le_)
        return ls, le, hi
