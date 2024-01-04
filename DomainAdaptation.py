from dataset_io import Dataset, MixedDataset
import config
import model
import train
import fire
from inference import infer_and_evaluate, save_plot
import torch
import numpy as np
from tqdm import tqdm
from torch import nn
import math
import copy
import os
from pathlib import Path

def DomainAdaptation(run_id=201, source_iters=10, target_iters=100, tag="DA", disable_jump=False):
    print(f"Running {tag} {run_id}...")
    config.domain_backprop = True
    M = model.prep_model(model.Net())
    optimizer_G = torch.optim.Adam(M.parameters(), lr=1e-5, betas=(0.9, 0.999))
    config.tags.append(tag)
    config.run_id = run_id
    run_id = f"{config.run_id:03d}"
    experiment_dir = os.path.join(config.experiments_dir, run_id)
    inference_dir = experiment_dir + "/inference/"
    Path(experiment_dir).mkdir(parents=True, exist_ok=True)
    source_ds = Dataset(config.source_dataset, config.source_var, "all")
    target_ds = Dataset(config.target_dataset, config.target_var, "train")


    criterion = nn.MSELoss()
    domain_criterion = nn.MSELoss()
    zeros = torch.zeros(config.batch_size).to(config.device)
    ones = torch.ones(config.batch_size).to(config.device)

    # check if source is already trained
    if os.path.exists(f"{config.experiments_dir}/{config.run_id:03d}/source_trained.pth") and not disable_jump:
        print("Source already trained")
        M.load_state_dict(torch.load(f"{experiment_dir}/source_trained.pth"))
    else:
        # Phase 1: Train on source
        config.set_status("Source Training")
        for source_iter in range(source_iters):
            print("-" * 20)
            print(f"Source Iteration: {source_iter + 1}/{source_iters}")
            source_data = source_ds.get_augmented_data()
            bp_crop_times = config.crop_times
            config.crop_times *= math.ceil(len(source_ds) / len(target_ds))
            target_data = iter(target_ds.get_augmented_data())
            config.crop_times = bp_crop_times
            vol_loss_total = label_correct = label_total_loss = 0
            for batch_idx, (low_res_source, high_res_source) in enumerate(tqdm(source_data, leave=False, desc="Source Training")):
                p = float((batch_idx + source_iter * len(source_ds)) / (source_iters * len(source_ds)))
                alpha = 2. / (1. + np.exp(-10 * p)) - 1
                optimizer_G.zero_grad()
                low_res_source = low_res_source.to(config.device)
                high_res_source = high_res_source.to(config.device)
                target_low = next(target_data)[0].to(config.device)
                pred_source, label_source = M(low_res_source[:, 0:1], low_res_source[:, -1:], alpha)
                _, label_target = M(target_low[:, 0:1], target_low[:, -1:], alpha)
                vol_loss = criterion(pred_source, high_res_source)
                label_loss = domain_criterion(label_source, zeros) + domain_criterion(label_target, ones)
                vol_loss_total += vol_loss.mean().item()
                label_correct += ((label_source < 0.5).sum() + (label_target > 0.5).sum()).item()
                label_total_loss += label_loss.mean().item()
                loss = vol_loss + label_loss
                loss.backward()
                optimizer_G.step()
            source_accuracy = label_correct / (len(source_ds) * 2)
            config.log({"Source Vol Loss": vol_loss_total, "Source Accuracy": source_accuracy, "Source Label Loss": label_total_loss})
        torch.save(M.state_dict(), f"{experiment_dir}/source_trained.pth")
        return
    # Evaluate source training efficiency
    # PSNR, PSNR_list = infer_and_evaluate(M, write_to_file=False, data=source_ds)
    # save_plot(PSNR, PSNR_list, config.experiments_dir + f"/{config.run_id:03d}", run_cycle=0)
    # Phase 2: Train on target
    config.set_status("Target Training")

    # lock feature extractors
    for mod in M.modules():
        if isinstance(mod, model.FeatureExtractor):
            mod.requires_grad_(False)
    for target_iter in range(target_iters):
        print("-" * 20)
        print(f"Target Iteration: {target_iter + 1}/{target_iters}")
        target_data = target_ds.get_augmented_data()
        vol_loss_total = 0
        for batch_idx, (low_res_source, high_res_source) in enumerate(tqdm(target_data, leave=False, desc="Target Training")):
            optimizer_G.zero_grad()
            low_res_source = low_res_source.to(config.device)
            high_res_source = high_res_source.to(config.device)
            pred_source, _ = M(low_res_source[:, 0:1], low_res_source[:, -1:])
            vol_loss = criterion(pred_source, high_res_source)
            vol_loss_total += vol_loss.mean().item()
            loss = vol_loss
            loss.backward()
            optimizer_G.step()
        config.log({"Source Vol Loss": vol_loss_total})
    PSNR, PSNR_list = infer_and_evaluate(M, write_to_file=True, inference_dir=inference_dir, experiments_dir=experiment_dir)
    save_plot(PSNR, PSNR_list, experiment_dir, run_cycle=1)
    torch.save(M.state_dict(), f"{experiment_dir}/target_trained.pth")

    config.set_status("Succeeded")


if __name__ == "__main__":
    fire.Fire(DomainAdaptation)
