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

def DomainAdaptation(run_id=220, source_iters=100, target_iters=100, tag="DA", load_model=True, stage="source"):
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
    source_ds = Dataset(config.source_dataset, config.source_var, "20")
    source_eval = Dataset(config.source_dataset, config.source_var, "all")
    target_ds = Dataset(config.target_dataset, config.target_var, "train")
    source_evaluate_every = 5
    target_evaluate_every = 20


    criterion = nn.MSELoss()
    domain_criterion = nn.L1Loss()

    # check if source is already trained
    if stage == "source":
        if os.path.exists(f"{experiment_dir}/source_trained.pth") and load_model:
            x = torch.load(f"{experiment_dir}/source_trained.pth")
            M = model.load_model(M, x)
            print("Model loaded")
        # Phase 1: Train on source
        config.set_status("Source Training")
        alpha_forward = 1
        alpha_reverse = 1e-3
        for source_iter in tqdm(range(source_iters),leave=False,desc="Source Training", position=0):
            tqdm.write("-" * 20)
            source_data = source_ds.get_augmented_data()
            bp_crop_times = config.crop_times
            config.crop_times *= math.ceil(len(source_ds) / len(target_ds))
            target_data = iter(target_ds.get_augmented_data())
            config.crop_times = bp_crop_times
            vol_loss_total = label_correct = label_total_loss = 0
            M.train()
            for batch_idx, (low_res_source, high_res_source) in enumerate(tqdm(source_data, leave=False, desc="Source Iters", position=1)):
                optimizer_G.zero_grad()
                # p = float((batch_idx + source_iter * len(source_ds)) / (source_iters * len(source_ds)))
                # alpha = 2. / (1. + np.exp(-10 * p)) - 1
                low_res_source = low_res_source.to(config.device)
                high_res_source = high_res_source.to(config.device)
                target_low = next(target_data)[0].to(config.device)
                # target_low = torch.zeros(config.batch_size, 2, *config.crop_size).to(config.device)

                pred_source, label_source = M(low_res_source[:, 0:1], low_res_source[:, -1:], alpha_forward, alpha_reverse)
                _, label_target = M(target_low[:, 0:1], target_low[:, -1:], alpha_forward, alpha_reverse)
                optimizer_G.zero_grad()

                vol_loss = criterion(pred_source, high_res_source)
                label_loss = domain_criterion(label_source, torch.zeros(label_source.size()).to(config.device)) + domain_criterion(label_target, torch.ones(label_target.size()).to(config.device))
                vol_loss_total += vol_loss.mean().item()
                label_correct += ((label_source <= 0.5).sum() + (label_target > 0.5).sum()).item()
                label_total_loss += label_loss.mean().item()
                loss = vol_loss + label_weight*label_loss
                # loss = label_loss
                # loss = vol_loss
                loss.backward()
                optimizer_G.step()
            source_accuracy = label_correct / (len(source_ds) * 2 * (config.interval+2))
            if label_total_loss/len(source_data)>0.99:
                for mod in next(iter(M.children())).children():
                    if isinstance(mod, model.DomainClassifier):
                        model.weight_reset(mod)
                alpha_reverse = 0
            else:
                alpha_reverse = (1-label_total_loss/len(source_data))*1e-3
            config.log({"Source Vol Loss": vol_loss_total/len(source_data), "Source Accuracy": source_accuracy, "Source Label Loss": label_total_loss/len(source_data), "Alpha Rev": alpha_reverse})
            torch.save(M.state_dict(), f"{experiment_dir}/source_trained.pth")
            if source_iter % source_evaluate_every == 1:
                PSNR_target, _ = infer_and_evaluate(M)
                PSNR_source, _ = infer_and_evaluate(M, data=source_eval)
                config.log({"Source PSNR": PSNR_source, "Target PSNR": PSNR_target})
    else:
        # Evaluate source training efficiency
        # PSNR, PSNR_list = infer_and_evaluate(M, write_to_file=False, data=source_ds)
        # save_plot(PSNR, PSNR_list, config.experiments_dir + f"/{config.run_id:03d}", run_cycle=0)
        # Phase 2: Train on target
        config.set_status("Target Training")
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
                pred_source, _ = M(low_res_source[:, 0:1], low_res_source[:, -1:])
                vol_loss = criterion(pred_source, high_res_source)
                vol_loss_total += vol_loss.mean().item()
                loss = vol_loss
                loss.backward()
                optimizer_G.step()
            config.log({"Target Vol Loss": vol_loss_total})
            torch.save(M.state_dict(), f"{experiment_dir}/target_trained.pth")
            if target_iter % target_evaluate_every == 1:
                PSNR, PSNR_list = infer_and_evaluate(M, write_to_file=True, inference_dir=inference_dir, experiments_dir=experiment_dir)
                save_plot(PSNR, PSNR_list, experiment_dir, run_cycle=1)

    config.set_status("Succeeded")


if __name__ == "__main__":
    fire.Fire(DomainAdaptation)
