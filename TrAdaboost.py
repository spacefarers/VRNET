from dataset_io import Dataset, MixedDataset
import config
import model
import train
import fire
from inference import infer_and_evaluate, save_plot
import torch
import numpy as np
from tqdm import tqdm


def TrAdaboost(run_id=200, boosting_iters=5, cycles=1, tag="TrA"):
    print(f"Running {tag} {run_id}...")
    config.tags.append(tag)
    config.run_id = run_id

    source_ds = Dataset(config.source_dataset, config.source_var, "all")
    target_ds = Dataset(config.target_dataset, config.target_var, "train")
    mixed_ds = MixedDataset(source_ds, target_ds)

    weights_source = np.ones((len(source_ds), 1)) / len(source_ds)
    weights_target = np.ones((len(target_ds), 1)) / len(target_ds)
    weights = np.concatenate((weights_source, weights_target), axis=0)

    M = model.prep_model(model.Net())

    bata = 1 / (1 + np.sqrt(2 * np.log(len(target_ds) / boosting_iters)))

    for cycle in range(1, cycles + 1):
        print(f"Cycle {cycle}/{cycles}:")
        for boosting_iter in range(1, boosting_iters):
            print(f"Boosting iteration {boosting_iter}/{boosting_iters}:")
            weights = balance_weights(weights)
            train_loader = mixed_ds.get_data(fixed=True)
            M = fit_model(M, train_loader, weights)
            error = calc_error(M, train_loader, weights)
            bata_T = error / (1 - error)

    config.set_status("Succeeded")


def weighted_MSE(pred, target, weights):
    return torch.mean(weights * (pred - target) ** 2)


def fit_model(model, train_loader, weights):
    for batch_idx, (low_res, high_res) in enumerate(tqdm(train_loader, desc="Fitting Model", leave=False)):
        low_res = low_res.to(config.device)
        high_res = high_res.to(config.device)
        weights = weights[batch_idx * config.batch_size: (batch_idx + 1) * config.batch_size]
        model.train()
        model.optimizer.zero_grad()
        pred, _ = model(low_res[:, 0:1], low_res[:, -1:])
        loss = weighted_MSE(pred, high_res, weights)
        loss.backward()
        model.optimizer.step()
    return model


def calc_error(model, train_loader, weights):
    error = []
    for batch_idx, (low_res, high_res) in enumerate(tqdm(train_loader, desc="Calculating error", leave=False)):
        low_res = low_res.to(config.device)
        high_res = high_res.to(config.device)
        weights = weights[batch_idx * config.batch_size: (batch_idx + 1) * config.batch_size]
        model.eval()
        pred, _ = model(low_res[:, 0:1], low_res[:, -1:])
        error.append(weighted_MSE(pred, high_res, weights))
    return error


def balance_weights(weights):
    total = np.sum(weights)
    return np.asarray(weights / total, order='C')


if __name__ == "__main__":
    fire.Fire(run)
