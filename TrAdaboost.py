from dataset_io import Dataset, MixedDataset
import config
import model
import fire
from inference import infer_and_evaluate, save_plot
import torch
import numpy as np
from tqdm import tqdm
from neptune.types import File
from torch import nn
import os
from matplotlib import pyplot as plt
from time import time

M = model.prep_model(model.Net())
optimizer_G = torch.optim.Adam(M.parameters(), lr=5e-5, betas=(0.9, 0.999))
stage = 2
source_len = 0


def TrAdaboost(run_id=200, boosting_iters=20, cycles=1, tag="TrA"):
    print(f"Running {tag} {run_id}...")
    config.tags.append(tag)
    config.run_id = run_id
    run_id = f"{config.run_id:03d}"
    experiment_dir = os.path.join(config.experiments_dir, run_id)
    inference_dir = experiment_dir + "/inference/"

    source_ds = Dataset(config.source_dataset, config.source_var, "all")
    global source_len, stage
    source_len = len(source_ds)
    target_ds = Dataset(config.target_dataset, config.target_var, "train")
    target_inference_ds = Dataset(config.target_dataset, config.target_var, "all")
    mixed_ds = MixedDataset([source_ds, target_ds])

    weights_source = np.ones((len(source_ds), 1))
    weights_target = np.ones((len(target_ds), 1))
    weights = np.concatenate((weights_source, weights_target), axis=0).squeeze(1)
    prev_error = np.zeros((len(source_ds) + len(target_ds)))

    bata = 1 / (1 + np.sqrt(2 * np.log(len(source_ds) / boosting_iters)))

    for cycle in range(1, cycles + 1):
        config.log({"Cycle": cycle})
        for boosting_iter in range(1, boosting_iters+1):
            print("-"*20)
            config.log({"Boosting Iteration": boosting_iter})
            start_time = time()
            print(f"stage: {stage}")
            weights = balance_weights(weights)
            train_loader = mixed_ds.get_data(fixed=True)
            fit_model(train_loader, weights)
            error = calc_error(train_loader)
            avg_target_error = np.mean(error[len(source_ds):])
            config.log({"Avg Target Error": avg_target_error})
            if avg_target_error > 0.1 and stage == 1:
                continue
            elif stage == 1:
                stage = 2
                continue
            config.log({"Total Error": np.sum(error)})
            error_diff = error - prev_error
            prev_error = error
            plt.clf()
            plt.plot(error_diff)
            plt.savefig(experiment_dir + f'/error_diff{(cycle - 1) * boosting_iters + boosting_iter}.png')
            # normalize error
            error = error / np.max(error)
            avg_target_error = weights[len(source_ds):].dot(error[len(source_ds):]) / len(target_ds)
            bata_T = avg_target_error / (1 - avg_target_error)
            for i in range(len(source_ds)):
                weights[i] = weights[i] * np.power(bata, error[i])
            for i in range(len(source_ds), len(source_ds) + len(target_ds)):
                weights[i] = weights[i] * np.power(bata_T, (-error[i]))
            end_time = time()
            config.log({"Time Cost": end_time - start_time})
            plt.clf()
            plt.plot(weights)
            plt.savefig(experiment_dir + f'/weights_{(cycle - 1) * boosting_iters + boosting_iter}.png')
            config.log({"weights": File(experiment_dir + f'/weights_{(cycle - 1) * boosting_iters + boosting_iter}.png')})
            source_to_target_weight_ratio = np.sum(weights[:len(source_ds)]) / np.sum(weights[len(source_ds):])
            config.log({"Source to Target Weight Ratio": source_to_target_weight_ratio})
        torch.save(M.state_dict(), experiment_dir + f'/finetune1.pth')
        PSNR, PSNR_list = infer_and_evaluate(M, write_to_file=True, inference_dir=inference_dir,
                                             experiments_dir=experiment_dir,data=target_inference_ds)
        save_plot(PSNR, PSNR_list, experiment_dir, run_cycle=cycle)

    config.set_status("Succeeded")


MSE = nn.MSELoss(reduction='none')
L1Loss = nn.L1Loss(reduction='none')


def unweighted_loss(pred, target):
    pred = pred.view(config.batch_size, -1)
    target = target.view(config.batch_size, -1)
    return torch.mean(L1Loss(pred, target), dim=1)


def fit_model(train_loader, weights):
    weights = torch.FloatTensor(weights).to(config.device)
    global M, optimizer_G
    for batch_idx, (low_res, high_res) in enumerate(tqdm(train_loader, desc="Fitting Model", leave=False)):
        if stage == 1 and batch_idx * config.batch_size < source_len:
            continue
        low_res = low_res.to(config.device)
        high_res = high_res.to(config.device)
        M.train()
        optimizer_G.zero_grad()
        pred, _ = M(low_res[:, 0:1], low_res[:, -1:])
        loss = torch.sum(
            weights[batch_idx * config.batch_size: (batch_idx + 1) * config.batch_size] * unweighted_loss(pred,
                                                                                                          high_res))
        loss.backward()
        optimizer_G.step()


def calc_error(train_loader):
    global M, stage, source_len
    error = []
    M.eval()
    for batch_idx, (low_res, high_res) in enumerate(tqdm(train_loader, desc="Calculating error", leave=False)):
        if stage == 1 and batch_idx * config.batch_size < source_len:
            error += [0] * config.batch_size
            continue
        with torch.no_grad():
            low_res = low_res.to(config.device)
            high_res = high_res.to(config.device)
            pred, _ = M(low_res[:, 0:1], low_res[:, -1:])
            loss = unweighted_loss(pred, high_res)
            loss = list(loss.detach().cpu().numpy())
            error += loss
    return np.array(error)


def balance_weights(weights):
    total = np.sum(weights)
    return np.asarray(weights / total, order='C')


if __name__ == "__main__":
    fire.Fire(TrAdaboost)
