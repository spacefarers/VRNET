import config
import time
import torch
from tqdm import tqdm
from pathlib import Path
import json
from matplotlib import pyplot as plt
import numpy as np
from neptune.types import File
from model import Net, prep_model, load_model
from dataset_io import Dataset


def infer_and_evaluate(model, inference_dir=None, write_to_file=False, experiments_dir=None, data=None):
    if type(model) == str:
        model = load_model(prep_model(Net()), (torch.load(experiments_dir + model)))
    if write_to_file:
        Path(inference_dir).mkdir(parents=True, exist_ok=True)
    model.eval()
    print('=======Inference========')
    config.set_status("Inferring")
    if data is None:
        data = Dataset(config.target_dataset, config.target_var, "all")
    start_time = time.time()
    PSNR_list = []
    for ind, (low_res_window, high_res_window) in enumerate(
            tqdm(data.get_raw_data(), desc=f"Inferring {config.target_dataset}-{config.target_var}",leave=False)):
        low_res_window = low_res_window.to(config.device)
        with torch.no_grad():
            pred, _ = model(low_res_window[:, 0:1], low_res_window[:, -1:])
            pred = pred.detach().cpu().numpy()
            for batch_num in range(pred.shape[0]):
                for j in range(0 if ind + batch_num == 0 else 1, config.interval + 2):
                    data = pred[batch_num][j]
                    data = np.asarray(data, dtype='<f')
                    data = data.flatten('F')
                    GT = high_res_window[batch_num][j]
                    GT = np.asarray(GT, dtype='<f')
                    GT = GT.flatten('F')
                    GT_range = GT.max() - GT.min()
                    MSE = np.mean((GT - data) ** 2)
                    PSNR = 20 * np.log10(GT_range) - 10 * np.log10(MSE)
                    PSNR_list.append(PSNR)
                    if write_to_file:
                        data.tofile(
                            f'{inference_dir}{config.target_dataset}-{config.target_var}-{ind * config.batch_size * (config.interval + 1) + batch_num * (config.interval + 1) + j + 1}.raw',
                            format='<f')
    end_time = time.time()
    time_cost = end_time - start_time
    print('Inference time cost', time_cost, 's')
    PSNR = np.mean(PSNR_list[config.train_data_split:])
    inference_logs = {"time_cost": time_cost, "PSNR": PSNR, "PSNR_list": PSNR_list}
    if write_to_file:
        with open(experiments_dir + '/inference.json', 'w') as f:
            json.dump(inference_logs, f, indent=4)
    config.log({"PSNR": PSNR})
    return PSNR, PSNR_list


def save_plot(PSNR, PSNR_list, save_path, ensemble_iter=None, run_cycle=None):
    if ensemble_iter is not None:
        desc = f'#{config.run_id}{f" E.{ensemble_iter}"}: {config.target_dataset} {config.target_var} PSNR'
        name = f'PSNR-E.{ensemble_iter}'
    elif run_cycle is not None:
        desc = f'#{config.run_id}{f" C.{run_cycle}"}: {config.target_dataset} {config.target_var} PSNR'
        name = f'PSNR-C.{run_cycle}'
    else:
        desc = f'#{config.run_id}: {config.target_dataset} {config.target_var} PSNR'
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
    config.log({"PSNR Plot": File(save_path + f'/{name}.png')})
