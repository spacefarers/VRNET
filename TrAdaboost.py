from dataset_io import Dataset, MixedDataset
import config
import model
import fire
from inference import infer_and_evaluate, save_plot
import torch
import numpy as np
from tqdm import tqdm
from copy import deepcopy

M = model.prep_model(model.Net())
optimizer_G = torch.optim.Adam(M.parameters(), lr=config.lr[0], betas=(0.9, 0.999))

def TrAdaboost(run_id=200, boosting_iters=5, cycles=1, tag="TrA"):
    print(f"Running {tag} {run_id}...")
    config.tags.append(tag)
    config.run_id = run_id

    source_ds = Dataset(config.source_dataset, config.source_var, "all")
    target_ds = Dataset(config.target_dataset, config.target_var, "train")
    mixed_ds = MixedDataset([source_ds, target_ds])

    weights_source = np.ones((len(source_ds), 1)) / len(source_ds)
    weights_target = np.ones((len(target_ds), 1)) / len(target_ds)
    weights = np.concatenate((weights_source, weights_target), axis=0).squeeze(1)

    bata = 1 / (1 + np.sqrt(2 * np.log(len(target_ds) / boosting_iters)))

    for cycle in range(1, cycles + 1):
        print(f"Cycle {cycle}/{cycles}:")
        for boosting_iter in range(1, boosting_iters):
            print(f"Boosting iteration {boosting_iter}/{boosting_iters}:")
            weights = balance_weights(weights)
            train_loader = mixed_ds.get_data(fixed=True)
            error = calc_error(train_loader, weights)
            fit_model(deepcopy(train_loader), weights)
            print(error)
            bata_T = error / (1 - error)

    config.set_status("Succeeded")


def weighted_MSE(pred, target, weights):
    pred = pred.view(config.batch_size, -1)
    target = target.view(config.batch_size, -1)
    return weights * torch.mean((pred - target) ** 2, dim=1)


def fit_model(train_loader, weights):
    weights = torch.FloatTensor(weights).to(config.device)
    global M, optimizer_G
    for batch_idx, (low_res, high_res) in enumerate(tqdm(train_loader, desc="Fitting Model", leave=False)):
        low_res = low_res.to(config.device)
        high_res = high_res.to(config.device)
        M.train()
        optimizer_G.zero_grad()
        pred, _ = M(low_res[:, 0:1], low_res[:, -1:])
        loss = torch.sum(weighted_MSE(pred, high_res, weights[batch_idx * config.batch_size: (batch_idx + 1) * config.batch_size]))
        loss.backward()
        optimizer_G.step()


def calc_error(train_loader, weights):
    weights = torch.FloatTensor(weights).to(config.device)
    global M
    error = []
    M.eval()
    for batch_idx, (low_res, high_res) in enumerate(tqdm(train_loader, desc="Calculating error", leave=False)):
        with torch.no_grad:
            low_res = low_res.to(config.device)
            high_res = high_res.to(config.device)
            pred, _ = M(low_res[:, 0:1], low_res[:, -1:])
            loss = weighted_MSE(pred, high_res, weights[batch_idx * config.batch_size: (batch_idx + 1) * config.batch_size])
            loss = list(loss.detach().cpu().numpy())
            error += loss
    return error


def balance_weights(weights):
    total = np.sum(weights)
    return np.asarray(weights / total, order='C')


if __name__ == "__main__":
    fire.Fire(TrAdaboost)
