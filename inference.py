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
    PSNR_list = []
    for ind, (low_res_window, high_res_window) in enumerate(tqdm(data.get_raw_data(), desc=f"Inferring {self.dataset.dataset}-{self.dataset.selected_var}")):
        low_res_window = low_res_window.to(config.device)
        high_res_window = high_res_window.to(config.device)
        low_res_window = low_res_window.unsqueeze(1)
        high_res_window = high_res_window.unsqueeze(1)
        with torch.no_grad():
            pred, _ = self.model(low_res_window[0], low_res_window[-1])
            pred = pred.detach().cpu().numpy()
            for batch_num in range(pred.shape[0]):
                for j in range(0 if ind+batch_num == 0 else 1, self.interval + 2):
                    data = pred[batch_num][j]
                    data = np.asarray(data, dtype='<f')
                    data = data.flatten('F')
                    GT = high_res_window[batch_num][j]
                    GT = GT.flatten('F')
                    GT_range = GT.max() - GT.min()
                    MSE = np.mean((GT - data) ** 2)
                    PSNR = 20 * np.log10(GT_range) - 10 * np.log10(MSE)
                    PSNR_list.append(PSNR)
                    if write_to_file:
                        data.tofile(
                            f'{self.inference_dir}{self.dataset.dataset}-{self.dataset.selected_var}-{ind * config.batch_size * (self.interval + 1) + batch_num * (self.interval + 1) + j + 1}.raw',
                            format='<f')
    end_time = time.time()
    time_cost = end_time - start_time
    print('Inference time cost', time_cost, 's')
    PSNR = np.mean(PSNR_list)
    inference_logs = {"time_cost": time_cost, "PSNR": PSNR, "PSNR_list": PSNR_list}
    self.inference_logs = inference_logs
    if write_to_file:
        with open(self.experiment_dir + '/inference.json', 'w') as f:
            json.dump(inference_logs, f, indent=4)
    return PSNR, PSNR_list


def save_plot(PSNR, PSNR_list, save_path, ensemble_iter=None, run_cycle=None):
    if ensemble_iter is not None:
        desc = f'#{config.run_id}{f" E.{self.ensemble_iter}"}: {config.dataset} {self.dataset.selected_var} PSNR'
        name = f'PSNR-E.{self.ensemble_iter}'
    elif run_cycle is not None:
        desc = f'#{config.run_id}{f" C.{self.run_cycle}"}: {config.dataset} {self.dataset.selected_var} PSNR'
        name = f'PSNR-C.{self.run_cycle}'
    else:
        desc = f'#{config.run_id}: {config.dataset} {self.dataset.selected_var} PSNR'
        name = 'PSNR'
    plt.clf()
    axes = plt.gca()
    axes.set_ylim([0, np.max(PSNR_list) + 5])
    plt.plot(PSNR_list)
    plt.axhline(y=PSNR, color='r', linestyle='--')
    plt.xlabel('Frame')
    plt.ylabel('PSNR')
    plt.title(desc)
    plt.yticks(list(plt.yticks()[0]) + [PSNR])
    plt.savefig(save_path + f'/{name}.png', dpi=300)
    config.log({name: wandb.Image(save_path + f'/{name}.png', caption=f'{desc}')})
