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

def DomainAdaptation(run_id=300, source_iters=100, target_iters=100, tag="DA", load_model=True, stage="source", use_restorer=True):
    print(f"Running {tag} {run_id}...")
    config.domain_backprop = False
    M = model.prep_model(model.Net())
    optimizer_G = torch.optim.Adam(M.parameters(), lr=1e-4, betas=(0.9, 0.999))
    config.tags.append(tag)
    config.run_id = run_id
    run_id = f"{config.run_id:03d}"
    experiment_dir = os.path.join(config.experiments_dir, run_id)
    inference_dir = experiment_dir + "/inference/"
    Path(experiment_dir).mkdir(parents=True, exist_ok=True)
    source_ds = Dataset(config.source_dataset, config.source_var, "all")
    target_ds = Dataset(config.target_dataset, config.target_var, "train")
    source_evaluate_every = 5
    target_evaluate_every = 20


    criterion = nn.MSELoss()
    domain_criterion = nn.L1Loss()

    # check if source is already trained
    if stage == "source" or stage == "all":
        config.enable_restorer = use_restorer
        if os.path.exists(f"{experiment_dir}/source_trained.pth") and load_model:
            x = torch.load(f"{experiment_dir}/source_trained.pth")
            M = model.load_model(M, x)
            print("Model loaded")
        # Phase 1: Train on source
        config.set_status("Source Training")
        for source_iter in tqdm(range(source_iters),leave=False,desc="Source Training", position=0):
            tqdm.write("-" * 20)
            source_data = source_ds.get_augmented_data()
            bp_crop_times = config.crop_times
            config.crop_times *= math.ceil(len(source_ds) / len(target_ds))
            target_data = iter(target_ds.get_augmented_data())
            config.crop_times = bp_crop_times
            vol_loss_total = target_restore_total_loss = 0
            M.train()
            for batch_idx, (low_res_source, high_res_source) in enumerate(tqdm(source_data, leave=False, desc="Source Iters", position=1)):
                optimizer_G.zero_grad()
                low_res_source = low_res_source.to(config.device)
                high_res_source = high_res_source.to(config.device)
                target_low = next(target_data)[0].to(config.device)

                optimizer_G.zero_grad()
                pred_source, _, _ = M(low_res_source[:, 0:1], low_res_source[:, -1:])
                loss = criterion(pred_source, high_res_source)
                vol_loss_total += loss.mean().item()
                if use_restorer:
                    _, _, target_restore = M(target_low[:, 0:1], target_low[:, -1:])
                    target_restore_loss = criterion(target_restore, target_low)
                    target_restore_total_loss += target_restore_loss.mean().item()
                    loss += target_restore_loss
                loss.backward()
                optimizer_G.step()
            config.log({"Source Vol Loss": vol_loss_total/len(source_data), "Source Target Restore Loss": target_restore_total_loss/len(source_data)})
            torch.save(M.state_dict(), f"{experiment_dir}/source_trained.pth")
            if source_iter % source_evaluate_every == 1:
                PSNR_target, _ = infer_and_evaluate(M)
                PSNR_source, _ = infer_and_evaluate(M, data=source_ds)
                config.log({"S1 Source PSNR": PSNR_source, "S1 Target PSNR": PSNR_target})
    if stage == "target" or stage == "all":
        config.enable_restorer = False
        # Evaluate source training efficiency
        # PSNR, PSNR_list = infer_and_evaluate(M, write_to_file=False, data=source_ds)
        # save_plot(PSNR, PSNR_list, config.experiments_dir + f"/{config.run_id:03d}", run_cycle=0)
        # Phase 2: Train on target
        config.set_status("Target Training")
        if stage == "target":
            M = model.load_model(M, torch.load(f"{experiment_dir}/{'target_trained' if load_model and os.path.exists(f'{experiment_dir}/target_trained.pth') else 'source_trained'}.pth"))
        # lock feature extractors
        # for mod in next(iter(M.children())).children():
        #     if isinstance(mod, model.FeatureExtractor):
        #         mod.requires_grad_(False)
            # if isinstance(mod, nn.Sequential) and not isinstance(mod[0], model.FeatureExtractor):
            #     model.weight_reset(mod)


        for target_iter in tqdm(range(target_iters), leave=False, desc="Target Training"):
            tqdm.write("-" * 20)
            target_data = target_ds.get_augmented_data()
            vol_loss_total = 0
            M.train()
            for batch_idx, (low_res_source, high_res_source) in enumerate(tqdm(target_data, leave=False, desc="Target Iters")):
                optimizer_G.zero_grad()
                low_res_source = low_res_source.to(config.device)
                high_res_source = high_res_source.to(config.device)
                pred_source, _, _ = M(low_res_source[:, 0:1], low_res_source[:, -1:])
                vol_loss = criterion(pred_source, high_res_source)
                vol_loss_total += vol_loss.mean().item()
                loss = vol_loss
                loss.backward()
                optimizer_G.step()
            config.log({"Target Vol Loss": vol_loss_total})
            torch.save(M.state_dict(), f"{experiment_dir}/target_trained.pth")
            if target_iter % target_evaluate_every == 1:
                # PSNR, PSNR_list = infer_and_evaluate(M, write_to_file=True, inference_dir=inference_dir, experiments_dir=experiment_dir)
                PSNR_target, _ = infer_and_evaluate(M)
                PSNR_source, _ = infer_and_evaluate(M, data=source_ds)
                config.log({"S2 Source PSNR": PSNR_source, "S2 Target PSNR": PSNR_target})

    config.set_status("Succeeded")


if __name__ == "__main__":
    fire.Fire(DomainAdaptation)
