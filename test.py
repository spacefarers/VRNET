from dataset_io import Dataset
import config
import model
import fire
from inference import infer_and_evaluate, save_plot
import torch
import numpy as np
from tqdm import tqdm
from torch import nn
import math
import os
from pathlib import Path

label_weight = 1


def DomainAdaptation(run_id=31, source_iters=100, target_iters=200, tag="DA", load_model=True, stage="target",
                     use_restorer=True):
    print(f"Running {tag} {run_id}...")
    config.domain_backprop = False
    config.tags.append(tag)
    config.run_id = run_id
    run_id = f"{config.run_id:03d}"
    experiment_dir = os.path.join(config.experiments_dir, run_id)
    inference_dir = experiment_dir + "/inference/"
    Path(experiment_dir).mkdir(parents=True, exist_ok=True)

    M = model.prep_model(model.Net())
    optimizer = torch.optim.Adam(M.parameters(), lr=5e-5, betas=(0.9, 0.999))
    model_load_path = None
    if stage == "source" or stage == "all":
        if os.path.exists(f"{experiment_dir}/source_trained.pth") and load_model:
            model_load_path = f"{experiment_dir}/source_trained.pth"
    else:
        if load_model and os.path.exists(f'{experiment_dir}/target_trained.pth'):
            model_load_path = f'{experiment_dir}/target_trained.pth'
        else:
            model_load_path = f'{experiment_dir}/source_trained.pth'
    if model_load_path is not None:
        print("Loading model from: ", model_load_path)
        M,optimizer = model.load_model(M, torch.load(model_load_path),optimizer)

    PSNR, PSNR_list = infer_and_evaluate(M)
    print(PSNR)


if __name__ == "__main__":
    fire.Fire(DomainAdaptation)
