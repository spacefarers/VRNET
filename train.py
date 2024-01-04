from tqdm import tqdm
from torch import nn, optim
import torch
import os
import numpy as np
from pathlib import Path
import json
import time
import config
from torch.nn import functional as F
from model import load_model

import dataset_io


class Trainer:
    def __init__(self, dataset, model, discriminator=None):
        self.model = model
        self.discriminator = discriminator
        self.dataset = dataset
        self.optimizer_G = optim.Adam(model.parameters(), lr=config.lr[0], betas=(0.9, 0.999))
        self.optimizer_D = optim.Adam(discriminator.parameters(), lr=config.lr[1], betas=(0.9, 0.999))
        self.criterion = nn.MSELoss()
        self.domain_criterion = nn.MSELoss()
        self.run_id = f"{config.run_id:03d}"
        self.stages = ['finetune1', 'finetune2']
        self.experiment_dir = os.path.join(config.experiments_dir, self.run_id)
        Path(self.experiment_dir).mkdir(parents=True, exist_ok=True)
        self.inference_dir = self.experiment_dir + "/inference/"
        self.predict_data = []
        self.inference_logs = None
        self.interval = config.interval
        self.ensemble_iter = config.ensemble_iter
        self.run_cycle = config.run_cycle

    def load_model(self, path):
        self.model = load_model(self.model, torch.load(path))

    def jump_to_progress(self):
        for stage in reversed(self.stages):
            if os.path.exists(self.experiment_dir + f'/{stage}.pth'):
                print('Jump to %s' % stage)
                self.load_model(self.experiment_dir + f'/{stage}.pth')
                return stage
        return None

    def save_model(self, stage, logs=None):
        torch.save(self.model.state_dict(), self.experiment_dir + f'/{stage}.pth')
        if logs:
            with open(self.experiment_dir + f'/{stage}_logs.json', 'w') as f:
                json.dump(logs, f, indent=4)

    def finetune1(self):
        time_start = time.time()
        finetune1_logs = {"loss": [], "domain_loss": [], "domain_accuracy": []}
        config.set_status("Finetune 1")
        source_dataset = None
        for epoch in tqdm(range(config.finetune1_epochs), position=0, leave=False):
            target_loader = self.dataset.get_augmented_data()
            train_length = len(target_loader)
            target_iter = iter(target_loader)
            total_loss = 0
            for batch_idx in tqdm(range(train_length), position=1, leave=False):
                self.optimizer_G.zero_grad()
                p = float((batch_idx + epoch * train_length) / (config.finetune1_epochs * train_length))
                alpha = 2. / (1. + np.exp(-10 * p)) - 1
                target_obj = next(target_iter)
                target_low, target_high = target_obj
                target_low = target_low.to(config.device)
                target_high = target_high.to(config.device)
                target_out, _ = self.model(target_low[:, 0:1], target_low[:, -1:], alpha)
                target_out_err = self.criterion(target_out, target_high)
                error = target_out_err
                total_loss += error.mean().item()
                error.backward()
                self.optimizer_G.step()
            finetune1_logs["loss"].append(total_loss)
            tqdm.write(f'FT1 loss: {total_loss}')
            config.log({"FT1 loss": total_loss})
        time_end = time.time()
        finetune1_time_cost = time_end - time_start
        finetune1_logs["time_cost"] = finetune1_time_cost
        torch.cuda.empty_cache()
        return finetune1_logs

    def finetune2(self):
        time_start = time.time()
        finetune2_logs = {"generator_loss": [], "discriminator_loss": [], "domain_loss": []}
        config.set_status("Finetune 2")
        for epoch in tqdm(range(1, config.finetune2_epochs + 1), position=0):
            train_loader = self.dataset.get_augmented_data()
            generator_loss = 0
            discriminator_loss = 0
            domain_total_loss = 0
            domain_acc = 0
            for batch_idx, (low_res_crops, high_res_crops, domain_label) in enumerate(tqdm(train_loader, position=1, leave=False)):
                p = float((batch_idx + epoch * len(train_loader)) / (config.finetune2_epochs * len(train_loader)))
                alpha = 2. / (1. + np.exp(-10 * p)) - 1

                low_res_crops = low_res_crops.to(config.device)
                high_res_crops = high_res_crops.to(config.device)

                for p in self.model.parameters():
                    p.requires_grad = False

                self.optimizer_D.zero_grad()
                output_real = self.discriminator(high_res_crops)
                label_real = torch.ones(output_real.size()).to(config.device)
                real_loss = self.criterion(output_real, label_real)
                fake_data, _ = self.model(low_res_crops[:,0:1], low_res_crops[:,-1:], alpha)
                label_fake = torch.zeros(output_real.size()).to(config.device)
                output_fake = self.discriminator(fake_data)
                fake_loss = self.criterion(output_fake, label_fake)
                loss = 0.5 * (real_loss + fake_loss)
                loss.backward()
                discriminator_loss += loss.mean().item()
                self.optimizer_D.step()

                for p in self.model.parameters():
                    p.requires_grad = True
                for p in self.discriminator.parameters():
                    p.requires_grad = False

                high, domain_class = self.model(low_res_crops[:,0:1], low_res_crops[:,-1:], alpha)
                output_real = self.discriminator(high)
                self.optimizer_G.zero_grad()
                label_real = torch.ones(output_real.size()).to(config.device)
                real_loss = self.criterion(output_real, label_real)
                error = (config.interval + 2) * self.criterion(high, high_res_crops) + 1e-3 * real_loss
                generator_loss += error.mean().item()
                if "ensemble_training" in config.tags and config.domain_backprop:
                    domain_label = F.one_hot(domain_label, num_classes=len(config.pretrain_vars)).float().to(config.device)
                    domain_loss = (config.interval + 2) * self.domain_criterion(domain_class, domain_label) / 10
                    domain_total_loss += domain_loss.mean().item()
                    error += domain_loss
                    domain_acc += torch.sum(torch.argmax(domain_class, dim=1) == torch.argmax(domain_label, dim=1)).detach().cpu().item()
                error.backward()
                self.optimizer_G.step()

                for p in self.discriminator.parameters():
                    p.requires_grad = True
            domain_acc /= (len(train_loader) * config.batch_size)
            if "ensemble_training" in config.tags and config.domain_backprop:
                finetune2_logs["domain_loss"].append(domain_total_loss)
                config.log({"FT2 domain loss": domain_total_loss})
                config.log({"FT2 domain accuracy": domain_acc * 100})
                tqdm.write(f'FT2 domain loss: {domain_total_loss}')
                tqdm.write(f'FT2 domain accuracy: {domain_acc * 100:.2f}%')
            finetune2_logs["generator_loss"].append(generator_loss)
            finetune2_logs["discriminator_loss"].append(discriminator_loss)
            tqdm.write(f'FT2 G loss: {generator_loss}, D loss: {discriminator_loss}')
            config.log({"FT2 G loss": generator_loss, "FT2 D loss": discriminator_loss})
        time_end = time.time()
        finetune2_time_cost = time_end - time_start
        print('FT2 time cost', finetune2_time_cost, 's')
        finetune2_logs["time_cost"] = finetune2_time_cost
        self.save_model('finetune2', finetune2_logs)
        torch.save(self.discriminator.state_dict(), self.experiment_dir + f'/ft2_D.pth')
        torch.cuda.empty_cache()
        return finetune2_logs

    def train(self, disable_jump=False):
        self.model.train()
        stage = self.jump_to_progress() if not disable_jump else None
        if (stage is None or stage == 'pretrain') and config.finetune1_epochs > 0:
            print('=======Finetune 1========')
            finetune1_logs = self.finetune1()
            self.save_model('finetune1', finetune1_logs)
            print('FT1 time cost', finetune1_logs["time_cost"], 's')
        if stage != 'finetune2' and config.finetune2_epochs > 0:
            print("=======Finetune 2========")
            finetune2_logs = self.finetune2()
            self.save_model('finetune2', finetune2_logs)
            print('FT2 time cost', finetune2_logs["time_cost"], 's')
        return self.model, self.discriminator
