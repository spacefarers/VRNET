import os
import numpy as np
import json
from skimage.transform import resize
from pathlib import Path
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

batch_size = 8


class Dataset:
    def __init__(self, root_data_dir, dataset, selected_var, scale):
        self.dataset = dataset
        self.root_data_dir = root_data_dir
        self.selected_var = selected_var
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

    def load(self):
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

    def spacial_dataloader(self, interval, crop_times):
        num_windows = len(self.hi_res) - interval - 1
        ls_train = np.zeros((crop_times * num_windows, 1, self.crop_size[0], self.crop_size[1], self.crop_size[2]))
        le_train = np.zeros((crop_times * num_windows, 1, self.crop_size[0], self.crop_size[1], self.crop_size[2]))
        li_train = np.zeros(
            (crop_times * num_windows, interval + 2, self.crop_size[0], self.crop_size[1], self.crop_size[2]))
        idx = 0
        for t in range(0, num_windows):
            ls_, le_, li_ = self.spacial_crop(self.low_res[t:t + interval + 2], crop_times, interval)
            for j in range(0, crop_times):
                ls_train[idx] = ls_[j]
                le_train[idx] = le_[j]
                li_train[idx] = li_[j]
                idx += 1
        ls_train = torch.FloatTensor(ls_train)
        le_train = torch.FloatTensor(le_train)
        li_train = torch.FloatTensor(li_train)
        data = torch.utils.data.TensorDataset(ls_train, le_train, li_train)
        train_loader = DataLoader(dataset=data, batch_size=batch_size, shuffle=True)
        return train_loader

    def spacial_crop(self, low_res_window, crop_times, interval):
        low_start = []
        low_end = []
        low_all = []
        n = 0
        while n < crop_times:
            x = np.random.randint(0, self.dims[0] // self.scale - self.crop_size[0])
            y = np.random.randint(0, self.dims[1] // self.scale - self.crop_size[1])
            z = np.random.randint(0, self.dims[2] // self.scale - self.crop_size[2])

            ls_ = low_res_window[0:1, x:x + self.crop_size[0], y:y + self.crop_size[1], z:z + self.crop_size[2]]
            le_ = low_res_window[interval + 1:interval + 2, x:x + self.crop_size[0], y:y + self.crop_size[1],
                  z:z + self.crop_size[2]]
            li_ = low_res_window[:, x:x + self.crop_size[0], y:y + self.crop_size[1], z:z + self.crop_size[2]]
            low_start.append(ls_)
            low_end.append(le_)
            low_all.append(li_)
            n = n + 1
        return low_start, low_end, low_all

    def spacial_temporal_dataloader(self, interval, crop_times):
        num = len(self.low_res) - interval - 1
        ls_train = np.zeros((crop_times * num, 1, self.crop_size[0], self.crop_size[1], self.crop_size[2]))
        le_train = np.zeros((crop_times * num, 1, self.crop_size[0], self.crop_size[1], self.crop_size[2]))
        hi_train = np.zeros(
            (crop_times * num, interval + 2, self.crop_size[0] * self.scale, self.crop_size[1] * self.scale,
             self.crop_size[2] * self.scale))
        idx = 0
        for t in range(0, num):
            ls_, le_, hi_ = self.spacial_temporal_crop(self.low_res[t:t + interval + 2],
                                                       self.hi_res[t:t + interval + 2], crop_times, interval)
            for j in range(0, crop_times):
                ls_train[idx] = ls_[j]
                le_train[idx] = le_[j]
                hi_train[idx] = hi_[j]
                idx += 1
        ls_train = torch.FloatTensor(ls_train)
        le_train = torch.FloatTensor(le_train)
        hi_train = torch.FloatTensor(hi_train)
        data = torch.utils.data.TensorDataset(ls_train, le_train, hi_train)
        train_loader = DataLoader(dataset=data, batch_size=batch_size, shuffle=True)
        return train_loader

    def spacial_temporal_crop(self, low_res_window, high_res_window, crop_times, interval):
        hi = []
        ls = []
        le = []
        n = 0
        while n < crop_times:
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
            n = n + 1
        return ls, le, hi

    def psnr(self, eval_source):
        if type(eval_source) != str:
            eval_source = eval_source.inference_dir
        if not os.path.exists(eval_source):
            return float('nan'), []
        if os.path.exists(eval_source+'PSNR.json'):
            with open(eval_source+'PSNR.json', 'r') as f:
                data = json.load(f)
            return data['mean'], data['array']
        print("=======Evaluating========")
        PSNR_list = []
        for ind, GT in enumerate(tqdm(self.hi_res)):
            cmp = np.fromfile(f'{eval_source}{self.dataset}-{self.selected_var}-{ind + 1}.raw', dtype='<f')
            GT = GT.flatten('F')
            GT_range = GT.max() - GT.min()
            MSE = np.mean((GT - cmp) ** 2)
            PSNR = 20 * np.log10(GT_range) - 10 * np.log10(MSE)
            PSNR_list.append(PSNR)
        print(f"PSNR is {np.mean(PSNR_list)}")
        print(f"array:\n {PSNR_list}")
        with open(f'{eval_source}PSNR.json', 'w') as f:
            json.dump({"mean": np.mean(PSNR_list), "array": PSNR_list}, f)
        return np.mean(PSNR_list), PSNR_list
