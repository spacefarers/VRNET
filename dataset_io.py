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
    def __init__(self, selected_var, train_all_data=False):
        self.dataset = config.dataset
        self.root_data_dir = config.root_data_dir
        self.selected_var = selected_var
        self.batch_size = config.batch_size
        self.interval = config.interval
        self.hi_res_data_dir = os.path.join(config.processed_dir, self.dataset, self.selected_var,
                                            'high_res/')
        self.lo_res_data_dir = os.path.join(config.processed_dir, self.dataset, self.selected_var,
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
        self.scale = config.scale
        self.crop_size = config.crop_size
        self.crop_times = config.crop_times

        self.hi_res = []
        self.low_res = []
        self.interval_splice = [i for i in range(0, self.total_samples, self.interval + 1)]
        if train_all_data:
            self.train_splice = [i for i in range(0, self.total_samples)]
        else:
            self.train_splice = [i for i in range(0, self.total_samples * config.train_data_split // 100)]

    def prepare_data(self):
        source_data = []
        for i in tqdm(range(1, self.total_samples + 1),leave=False,desc=f"Reading data {self.dataset}-{self.selected_var}"):
            data = np.fromfile(f"{self.var_dir}{self.dataset}-{self.selected_var}-{i}.raw", dtype='<f')
            source_data.append(data)
        source_data = np.asarray(source_data)
        data_max = np.max(source_data)
        data_min = np.min(source_data)
        source_data = 2 * (source_data - data_min) / (data_max - data_min) - 1
        for i in tqdm(range(self.total_samples),desc=f"Writing to files",leave=False):
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

    def load(self):
        if len(self.hi_res) != 0:
            return
        if not self.check_processed_data():
            self.prepare_data()
        for i in tqdm(range(1, self.total_samples + 1), desc=f"Loading {self.dataset}-{self.selected_var}",leave=False):
            hi_ = np.fromfile(f'{self.hi_res_data_dir}{self.dataset}-{self.selected_var}-{i}.raw', dtype='<f')
            hi_ = hi_.reshape(self.dims[2], self.dims[1], self.dims[0]).transpose()
            lo_ = np.fromfile(f'{self.lo_res_data_dir}{self.dataset}-{self.selected_var}-{i}.raw', dtype='<f')
            lo_ = lo_.reshape(self.dims[2] // self.scale, self.dims[1] // self.scale,
                              self.dims[0] // self.scale).transpose()
            self.hi_res.append(hi_)
            self.low_res.append(lo_)
        self.hi_res = np.asarray(self.hi_res)
        self.low_res = np.asarray(self.low_res)

    def get_data(self, splice_strategy):
        self.load()
        if splice_strategy == "interval":
            splice_strategy = self.interval_splice
        elif splice_strategy == "train":
            splice_strategy = self.train_splice
        elif splice_strategy == "inference":
            lo_res_start = torch.FloatTensor(np.take(self.low_res, self.interval_splice[:-1], axis=0))
            lo_res_end = torch.FloatTensor(np.take(self.low_res, self.interval_splice[1:], axis=0))
            data = torch.utils.data.TensorDataset(lo_res_start, lo_res_end)
            inference_loader = DataLoader(dataset=data, batch_size=self.batch_size, shuffle=False)
            return inference_loader
        else:
            raise ValueError("Invalid splice strategy")
        num_windows = len(splice_strategy) - self.interval - 1
        lo_res_start = np.zeros(
            (self.crop_times * num_windows, 1, self.crop_size[0], self.crop_size[1], self.crop_size[2]))
        lo_res_end = np.zeros(
            (self.crop_times * num_windows, 1, self.crop_size[0], self.crop_size[1], self.crop_size[2]))
        lo_res_full = np.zeros(
            (self.crop_times * num_windows, self.interval + 2, self.crop_size[0], self.crop_size[1], self.crop_size[2]))
        hi_res_full = np.zeros(
            (
                self.crop_times * num_windows, self.interval + 2, self.crop_size[0] * self.scale,
                self.crop_size[1] * self.scale,
                self.crop_size[2] * self.scale))
        idx = 0
        for t in range(num_windows):
            ls_, le_, lf_, hf_ = self.random_crop_data(
                np.take(self.low_res, splice_strategy[t:t + self.interval + 2], axis=0),
                np.take(self.hi_res, splice_strategy[t:t + self.interval + 2], axis=0))
            for j in range(0, self.crop_times):
                lo_res_start[idx] = ls_[j]
                lo_res_end[idx] = le_[j]
                lo_res_full[idx] = lf_[j]
                hi_res_full[idx] = hf_[j]
                idx += 1
        lo_res_start = torch.FloatTensor(lo_res_start)
        lo_res_end = torch.FloatTensor(lo_res_end)
        lo_res_full = torch.FloatTensor(lo_res_full)
        hi_res_full = torch.FloatTensor(hi_res_full)
        # domain label repeat self.selected_var num_windows*crop_times times
        domain_label = torch.LongTensor(np.repeat(config.pretrain_vars.index(self.selected_var) if config.domain_backprop and 'ensemble_training' in config.tags else 0, num_windows * self.crop_times))
        data = torch.utils.data.TensorDataset(lo_res_start, lo_res_end, lo_res_full, hi_res_full, domain_label)
        train_loader = DataLoader(dataset=data, batch_size=self.batch_size, shuffle=True)
        return train_loader

    def random_crop_data(self, low_res_window, high_res_window):
        hi = []
        ls = []
        le = []
        li = []
        for _ in range(self.crop_times):
            x = np.random.randint(0, self.dims[0] // self.scale - self.crop_size[0])
            y = np.random.randint(0, self.dims[1] // self.scale - self.crop_size[1])
            z = np.random.randint(0, self.dims[2] // self.scale - self.crop_size[2])

            ls_ = low_res_window[0:1, x:x + self.crop_size[0], y:y + self.crop_size[1], z:z + self.crop_size[2]]
            le_ = low_res_window[self.interval + 1:self.interval + 2, x:x + self.crop_size[0], y:y + self.crop_size[1],
                  z:z + self.crop_size[2]]
            li_ = low_res_window[:, x:x + self.crop_size[0], y:y + self.crop_size[1], z:z + self.crop_size[2]]
            hi_ = high_res_window[:, x * self.scale:(x + self.crop_size[0]) * self.scale,
                  self.scale * y:(y + self.crop_size[1]) * self.scale,
                  self.scale * z:(z + self.crop_size[2]) * self.scale]
            ls.append(ls_)
            hi.append(hi_)
            li.append(li_)
            le.append(le_)
        return ls, le, li, hi

class MixedDataset:
    def __init__(self, datasets):
        self.datasets = datasets

    def get_data(self, splice_strategy):
        # combine 3 dataloaders
        dataloaders = [dataset.get_data(splice_strategy) for dataset in self.datasets]
        concat_loader = DataLoader(dataset=ConcatDataset([d.dataset for d in dataloaders]), batch_size=config.batch_size, shuffle=True)
        return concat_loader
