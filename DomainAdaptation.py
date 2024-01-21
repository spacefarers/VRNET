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

def DomainAdaptation(run_id=400, source_iters=100, target_iters=100, tag="DA", load_model=False, stage="source", use_restorer=True):
    print(f"Running {tag} {run_id}...")
    config.domain_backprop = False
    M = model.prep_model(model.Net())
    E = model.prep_model(M.module.encoder)
    U = model.prep_model(M.module.upscaler)
    R = model.prep_model(M.module.restorer)
    LRDC = model.prep_model(M.module.LR_domain_classifier)
    FDC = model.prep_model(M.module.feature_domain_classifier)
    optimizer = torch.optim.Adam(M.parameters(), lr=5e-5, betas=(0.9, 0.999))
    config.tags.append(tag)
    config.run_id = run_id
    run_id = f"{config.run_id:03d}"
    experiment_dir = os.path.join(config.experiments_dir, run_id)
    inference_dir = experiment_dir + "/inference/"
    Path(experiment_dir).mkdir(parents=True, exist_ok=True)
    source_ds = Dataset(config.source_dataset, config.source_var, "all")
    # eval_source_ds = Dataset(config.source_dataset, config.source_var, "all")
    eval_source_ds = source_ds
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
        for source_iter in tqdm(range(source_iters),leave=False,desc="Source Training", position=0):
            config.set_status("Source Training")
            tqdm.write("-" * 20)
            source_data = source_ds.get_augmented_data()
            bp_crop_times = config.crop_times
            config.crop_times *= math.ceil(len(source_ds) / len(target_ds))
            target_data = iter(target_ds.get_augmented_data())
            config.crop_times = bp_crop_times
            M.train()
            for batch_idx, (low_res_source, high_res_source) in enumerate(tqdm(source_data, leave=False, desc="Source Iters", position=1)):
                low_res_source = low_res_source.to(config.device)
                high_res_source = high_res_source.to(config.device)
                target_low, target_high = next(target_data)
                target_low = target_low.to(config.device)
                # target_high = target_high.to(config.device)

                # # Train Discriminators
                # M.encoder.requires_grad_(False)
                # M.restorer.requires_grad_(False)
                # M.LR_domain_classifier.requires_grad_(True)
                # M.feature_domain_classifier.requires_grad_(True)
                # optimizer.zero_grad()
                # features_S = M.encoder(low_res_source[:, 0:1], low_res_source[:, -1:])
                # features_T = M.encoder(target_low[:, 0:1], target_low[:, -1:])
                # feature_source_label = M.feature_domain_classifier(features_S)
                # feature_target_label = M.feature_domain_classifier(features_T)
                # features_label_loss = domain_criterion(feature_source_label, torch.ones_like(feature_source_label)) + domain_criterion(feature_target_label, torch.zeros_like(feature_target_label))
                # source_restore = M.restorer(features_S)
                # # target_restore = M.restorer(features_T)
                # LR_target_label = M.LR_domain_classifier(target_low)
                # LR_source_label = M.LR_domain_classifier(source_restore)
                # LR_label_loss = domain_criterion(LR_target_label, torch.ones_like(LR_target_label)) + domain_criterion(LR_source_label, torch.zeros_like(LR_source_label))
                # loss = LR_label_loss + features_label_loss
                # config.track({"S1 LRD Label Loss": LR_label_loss, "S1 FeatureD Label Loss": features_label_loss})
                # loss.backward()
                # optimizer.step()

                # Train Generators
                E.requires_grad_(True)
                R.requires_grad_(True)
                LRDC.requires_grad_(False)
                FDC.requires_grad_(False)
                optimizer.zero_grad()
                features_S = E(low_res_source[:, 0:1], low_res_source[:, -1:])
                features_T = E(target_low[:, 0:1], target_low[:, -1:])
                source_hi = U(features_S)
                # target_hi = U(features_T)
                # feature_source_label = FDC(features_S)
                # feature_target_label = FDC(features_T)
                # features_label_loss = domain_criterion(feature_source_label, torch.full_like(feature_source_label, 0.5)) + domain_criterion(feature_target_label, torch.full_like(feature_target_label, 0.5))
                # source_restore = R(features_S)
                target_restore = R(features_T)
                # LR_target_label = LRDC(target_low)
                # LR_source_label = LRDC(source_restore)
                # LR_label_loss = domain_criterion(LR_target_label, torch.full_like(LR_target_label, 0.5)) + domain_criterion(LR_source_label, torch.full_like(LR_source_label, 0.5))
                restore_loss = criterion(target_low, target_restore)
                # cycle_source_feature = E(source_restore[:, 0:1], source_restore[:, -1:])
                # cycle_source_hi = U(cycle_source_feature)
                # cycle_vol_loss = criterion(cycle_source_hi, high_res_source)
                vol_loss = criterion(source_hi, high_res_source)
                # vol_loss_target = criterion(target_hi, target_high)
                # source_identity_loss = criterion(cycle_source_feature, features_S)
                # loss = 0.01*LR_label_loss + 0.01*features_label_loss + 0.01*source_identity_loss + vol_loss + cycle_vol_loss + restore_loss
                # loss = 0.01*source_identity_loss + vol_loss + cycle_vol_loss + restore_loss
                loss = vol_loss + restore_loss
                # config.track({"S1 LR Label Loss": LR_label_loss, "S1 Feature Label Loss": features_label_loss, "S1 Source Identity Loss": source_identity_loss, "S1 Vol Loss": vol_loss, "S1 Cycle Vol Loss": cycle_vol_loss, "S1 Restore Loss": restore_loss})
                # config.track({"S1 Source Identity Loss": source_identity_loss, "S1 Vol Loss": vol_loss, "S1 Cycle Vol Loss": cycle_vol_loss, "S1 Restore Loss": restore_loss})
                config.track({"S1 Vol Loss": vol_loss, "S1 Restore Loss": restore_loss})
                loss.backward()
                optimizer.step()
            config.log_all()
            torch.save(M.state_dict(), f"{experiment_dir}/source_trained.pth")
            if source_iter % source_evaluate_every == source_evaluate_every - 1:
                PSNR_target, _ = infer_and_evaluate(M)
                PSNR_source, _ = infer_and_evaluate(M, data=eval_source_ds)
                config.log({"S1 Source PSNR": PSNR_source, "S1 Target PSNR": PSNR_target})
    if stage == "target" or stage == "all":
        config.enable_restorer = False
        # Evaluate source training efficiency
        # PSNR, PSNR_list = infer_and_evaluate(M, write_to_file=False, data=source_ds)
        # save_plot(PSNR, PSNR_list, config.experiments_dir + f"/{config.run_id:03d}", run_cycle=0)
        # Phase 2: Train on target
        if stage == "target":
            M = model.load_model(M, torch.load(f"{experiment_dir}/{'target_trained' if load_model and os.path.exists(f'{experiment_dir}/target_trained.pth') else 'source_trained'}.pth"))
        # lock feature extractors
        # for mod in next(iter(M.children())).children():
        #     if isinstance(mod, model.FeatureExtractor):
        #         mod.requires_grad_(False)
            # if isinstance(mod, nn.Sequential) and not isinstance(mod[0], model.FeatureExtractor):
            #     model.weight_reset(mod)


        for target_iter in tqdm(range(target_iters), leave=False, desc="Target Training"):
            config.set_status("Target Training")
            tqdm.write("-" * 20)
            target_data = target_ds.get_augmented_data()
            vol_loss_total = 0
            M.train()
            for batch_idx, (low_res_source, high_res_source) in enumerate(tqdm(target_data, leave=False, desc="Target Iters")):
                optimizer.zero_grad()
                low_res_source = low_res_source.to(config.device)
                high_res_source = high_res_source.to(config.device)
                pred_source, _, _ = M(low_res_source[:, 0:1], low_res_source[:, -1:])
                vol_loss = criterion(pred_source, high_res_source)
                vol_loss_total += vol_loss.mean().item()
                loss = vol_loss
                loss.backward()
                optimizer.step()
            config.log({"S2 Vol Loss": vol_loss_total})
            torch.save(M.state_dict(), f"{experiment_dir}/target_trained.pth")
            if target_iter % target_evaluate_every == target_evaluate_every - 1:
                # PSNR, PSNR_list = infer_and_evaluate(M, write_to_file=True, inference_dir=inference_dir, experiments_dir=experiment_dir)
                PSNR_target, _ = infer_and_evaluate(M)
                PSNR_source, _ = infer_and_evaluate(M, data=eval_source_ds)
                config.log({"S2 Source PSNR": PSNR_source, "S2 Target PSNR": PSNR_target})

    config.set_status("Succeeded")


if __name__ == "__main__":
    fire.Fire(DomainAdaptation)
