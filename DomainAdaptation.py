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


def DomainAdaptation(run_id=40, source_iters=50, target_iters=100, tag="DA", load_model=False, stage="all",
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
    if model_load_path is not None and os.path.exists(model_load_path):
        config.log({"Model": "Loaded"})
        print("Loading model from: ", model_load_path)
        M,optimizer = model.load_model(M, torch.load(model_load_path),optimizer)
    else:
        M.apply(model.weights_init_kaiming)
        config.log({"Model": "Not Loaded"})
    E = model.prep_model(M.module.encoder)
    U = model.prep_model(M.module.upscaler)
    R = model.prep_model(M.module.restorer)
    # LRDC = model.prep_model(M.module.LR_domain_classifier)
    # FDC = model.prep_model(M.module.feature_domain_classifier)

    source_ds = Dataset(config.source_dataset, config.source_var, "all")
    # eval_source_ds = Dataset(config.source_dataset, config.source_var, "all")
    eval_source_ds = source_ds
    target_ds = Dataset(config.target_dataset, config.target_var, "train")
    # source_aid_ds = Dataset("hurricane", "RAIN", "all")
    source_aid_ds = target_ds
    source_evaluate_every = 5
    target_evaluate_every = 20

    criterion = nn.MSELoss()
    domain_criterion = nn.L1Loss()

    # check if source is already trained
    if stage == "source" or stage == "all":
        config.enable_restorer = use_restorer
        # Phase 1: Train on source
        for source_iter in tqdm(range(source_iters), leave=False, desc="Source Training", position=0):
            config.set_status("Source Training")
            # tqdm.write("-" * 20)
            source_data = source_ds.get_augmented_data()
            bp_crop_times = config.crop_times
            config.crop_times *= math.ceil(len(source_ds) / len(source_aid_ds))
            target_data = iter(source_aid_ds.get_augmented_data())
            config.crop_times = bp_crop_times
            M.train()
            for batch_idx, (low_res_source, high_res_source) in enumerate(
                    tqdm(source_data, leave=False, desc="Source Iters", position=1)):
                low_res_source = low_res_source.to(config.device)
                high_res_source = high_res_source.to(config.device)
                target_low, target_high = next(target_data)
                target_low = target_low.to(config.device)
                # target_high = target_high.to(config.device)

                # Train Generators
                # E.requires_grad_(True)
                # R.requires_grad_(True)
                # LRDC.requires_grad_(False)
                # FDC.requires_grad_(False)
                optimizer.zero_grad()
                features_S = E(low_res_source[:, 0:1], low_res_source[:, -1:])
                features_T = E(target_low[:, 0:1], target_low[:, -1:])
                source_hi = U(features_S)
                vol_loss = criterion(source_hi, high_res_source)
                target_restore = R(features_T)
                restore_loss = criterion(target_restore, target_low)
                loss = vol_loss + restore_loss
                config.track({"S1 Vol Loss": vol_loss, "S1 Restore Loss": restore_loss})
                loss.backward()
                optimizer.step()
            config.log_all()
            if source_iter % source_evaluate_every == source_evaluate_every - 1:
                PSNR_target, _ = infer_and_evaluate(M)
                # PSNR_source, _ = infer_and_evaluate(M, data=eval_source_ds)
                config.log({"S1 Target PSNR": PSNR_target})
                model.save_model(M,optimizer, f"{experiment_dir}/source_trained.pth")
    if stage == "target" or stage == "all":
        config.enable_restorer = False
        # Evaluate source training efficiency
        # PSNR, PSNR_list = infer_and_evaluate(M, write_to_file=False, data=source_ds)
        # save_plot(PSNR, PSNR_list, config.experiments_dir + f"/{config.run_id:03d}", run_cycle=0)
        # Phase 2: Train on target
        # reset upscaler weights
        encoder_disable = True
        # model.weight_reset(M.module.upscaler)
        for stage in range(6):
            for target_iter in tqdm(range(target_iters), leave=False, desc=f"Target Training stage {stage}"):
                M.module.encoder.requires_grad_(not encoder_disable)
                encoder_disable = not encoder_disable
                config.set_status("Target Training")
                # tqdm.write("-" * 20)
                target_data = target_ds.get_augmented_data()
                for batch_idx, (low_res_source, high_res_source) in enumerate(
                        tqdm(target_data, leave=False, desc="Target Iters")):
                    optimizer.zero_grad()
                    low_res_source = low_res_source.to(config.device)
                    high_res_source = high_res_source.to(config.device)
                    pred_source, _, _ = M(low_res_source[:, 0:1], low_res_source[:, -1:])
                    vol_loss = criterion(pred_source, high_res_source)
                    loss = vol_loss
                    config.track({"S2 Vol Loss": vol_loss})
                    loss.backward()
                    optimizer.step()
                config.log_all()
                if target_iter % target_evaluate_every == target_evaluate_every - 1:
                    # PSNR, PSNR_list = infer_and_evaluate(M, write_to_file=True, inference_dir=inference_dir, experiments_dir=experiment_dir)
                    PSNR_target, PSNR_target_list = infer_and_evaluate(M)
                    # PSNR_source, _ = infer_and_evaluate(M, data=eval_source_ds)
                    try:
                        save_plot(PSNR_target, PSNR_target_list)
                    except:
                        print("Failed to save plot")
                    config.log({"S2 Target PSNR": PSNR_target})
                    model.save_model(M,optimizer, f"{experiment_dir}/target_trained.pth")

    config.set_status("Succeeded")


if __name__ == "__main__":
    fire.Fire(DomainAdaptation)
