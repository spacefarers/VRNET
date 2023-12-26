import config
import time
import torch
from tqdm import tqdm
from pathlib import Path
import json
import os
from matplotlib import pyplot as plt
import numpy as np
import wandb
from model import Net,prep_model
from dataset_io import Dataset

def infer_and_evaluate(self, model, write_to_file=False):
    if type(model) == str:
        model = prep_model(Net()).load_state_dict(torch.load(self.experiment_dir + f'/{model}.pth'))
    if write_to_file:
        Path(self.inference_dir).mkdir(parents=True, exist_ok=True)
    model.eval()
    print('=======Inference========')
    config.log({"status": 3})
    data = Dataset(self.dataset.dataset, self.dataset.selected_var, "all")
    start_time = time.time()
    for ind in tqdm(range(data.total_samples), desc=f"Inferring {self.dataset.dataset}-{self.dataset.selected_var}"):
        ls = data.low_res[ind].to(config.device)
        le = data.low_res[ind + config.interval+1].to(config.device)
        ls = ls.unsqueeze(1)
        le = le.unsqueeze(1)
        with torch.no_grad():
            pred, _ = self.model(ls, le)
            pred = pred.detach().cpu().numpy()
            for b in range(pred.shape[0]):
                for j in range(0 if b + ind == 0 else 1, self.interval + 2):
                    data = pred[b][j]
                    data = np.asarray(data, dtype='<f')
                    data = data.flatten('F')
                    if write_to_file:
                        data.tofile(
                            f'{self.inference_dir}{self.dataset.dataset}-{self.dataset.selected_var}-{ind * config.batch_size * (self.interval + 1) + b * (self.interval + 1) + j + 1}.raw',
                            format='<f')
    end_time = time.time()
    time_cost = end_time - start_time
    print('Inference time cost', time_cost, 's')
    PSNR, PSNR_list = self.psnr()
    inference_logs = {"time_cost": time_cost, "PSNR": PSNR, "PSNR_list": PSNR_list}
    self.inference_logs = inference_logs
    if write_to_file:
        with open(self.experiment_dir + '/inference.json', 'w') as f:
            json.dump(inference_logs, f, indent=4)
    return PSNR, PSNR_list


def psnr(self):
    print("=======Evaluating========")
    PSNR_list = []
    for ind, cmp in enumerate(tqdm(self.predict_data)):
        GT = self.dataset.hi_res[ind]
        GT = GT.flatten('F')
        GT_range = GT.max() - GT.min()
        MSE = np.mean((GT - cmp) ** 2)
        PSNR = 20 * np.log10(GT_range) - 10 * np.log10(MSE)
        PSNR_list.append(PSNR)
    print(f"PSNR is {np.mean(PSNR_list)}")
    print(f"array:\n {PSNR_list}")
    return np.mean(PSNR_list), PSNR_list


def save_plot(self):
    if self.ensemble_iter is not None:
        desc = f'#{config.run_id}{f" E.{self.ensemble_iter}"}: {config.dataset} {self.dataset.selected_var} PSNR'
        name = f'PSNR-E.{self.ensemble_iter}'
    elif self.run_cycle is not None:
        desc = f'#{config.run_id}{f" C.{self.run_cycle}"}: {config.dataset} {self.dataset.selected_var} PSNR'
        name = f'PSNR-C.{self.run_cycle}'
    else:
        desc = f'#{config.run_id}: {config.dataset} {self.dataset.selected_var} PSNR'
        name = 'PSNR'
    x = self.inference_logs["PSNR_list"]
    plt.clf()
    axes = plt.gca()
    axes.set_ylim([0, np.max(x) + 5])
    plt.plot(x)
    plt.axhline(y=self.inference_logs["PSNR"], color='r', linestyle='--')
    plt.xlabel('Frame')
    plt.ylabel('PSNR')
    plt.title(desc)
    plt.yticks(list(plt.yticks()[0]) + [self.inference_logs["PSNR"]])
    plt.savefig(self.experiment_dir + f'/{name}.png', dpi=300)
    config.log({name: wandb.Image(self.experiment_dir + f'/{name}.png', caption=f'{desc}')})
