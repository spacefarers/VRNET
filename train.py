from tqdm import tqdm
from torch import nn, optim
import torch
import os
import numpy as np
from pathlib import Path
import json
import time
import config
from matplotlib import pyplot as plt
import wandb
from torch.nn import functional as F
from dataset_io import Dataset, MixedDataset


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
        self.model.load_state_dict(torch.load(path))

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

    def train(self, disable_jump=False):
        pretrain_time_cost = 0
        finetune1_time_cost = 0
        finetune2_time_cost = 0
        self.model.train()
        stage = self.jump_to_progress() if not disable_jump else None
        if (stage is None or stage == 'pretrain') and config.finetune1_epochs > 0:
            # Finetune 1
            print('=======Finetune 1========')
            time_start = time.time()
            finetune1_logs = {"loss": [], "domain_loss": [], "domain_accuracy": []}
            config.log({"status": 1})
            for epoch in tqdm(range(1, config.finetune1_epochs + 1), position=0):
                train_loader = self.dataset.get_data("train")
                loss_mse = 0
                domain_total_loss = 0
                domain_acc = 0
                for batch_idx, (ls, le, li, hi, domain_label) in enumerate(tqdm(train_loader, position=1, leave=False)):
                    p = float((batch_idx + epoch * len(train_loader)) / (config.finetune1_epochs * len(train_loader)))
                    alpha = 2. / (1. + np.exp(-10 * p)) - 1
                    ls = ls.to(config.device)
                    le = le.to(config.device)
                    hi = hi.to(config.device)
                    high, domain_class = self.model(ls, le, alpha)
                    self.optimizer_G.zero_grad()
                    error = (config.interval + 2) * self.criterion(high, hi)
                    loss_mse += error.mean().item()
                    if "ensemble_training" in config.tags and config.domain_backprop:
                        domain_label = F.one_hot(domain_label, num_classes=len(config.pretrain_vars)).float().to(config.device)
                        domain_loss = (config.interval + 2) * self.domain_criterion(domain_class, domain_label) / 10
                        domain_total_loss += domain_loss.mean().item()
                        error += domain_loss
                        domain_acc += torch.sum(torch.argmax(domain_class, dim=1) == torch.argmax(domain_label,dim=1)).detach().cpu().item()
                    error.backward()
                    self.optimizer_G.step()
                domain_acc /= (len(train_loader) * config.batch_size)
                finetune1_logs["loss"].append(loss_mse)
                finetune1_logs["domain_loss"].append(domain_total_loss)
                finetune1_logs["domain_accuracy"].append(domain_acc)
                tqdm.write(f'FT1 loss: {loss_mse}')
                tqdm.write(f'FT1 domain loss: {domain_total_loss}')
                tqdm.write(f'FT1 domain accuracy: {domain_acc*100:.2f}%')
                config.log({"FT1 loss": loss_mse})
                config.log({"FT1 domain loss": domain_total_loss})
                config.log({"FT1 domain accuracy": domain_acc*100})
            time_end = time.time()
            finetune1_time_cost = time_end - time_start
            print('FT1 time cost', finetune1_time_cost, 's')
            finetune1_logs["time_cost"] = finetune1_time_cost
            self.save_model('finetune1', finetune1_logs)
            torch.cuda.empty_cache()

        if stage != 'finetune2' and config.finetune2_epochs > 0:
            # Finetune 2
            tqdm.write('=======Finetune 2========')
            time_start = time.time()
            finetune2_logs = {"generator_loss": [], "discriminator_loss": [], "domain_loss": []}
            config.log({"status": 2})
            for epoch in tqdm(range(1, config.finetune2_epochs + 1), position=0):
                train_loader = self.dataset.get_data("train")
                generator_loss = 0
                discriminator_loss = 0
                domain_total_loss = 0
                domain_acc = 0
                for batch_idx, (ls, le, li, hi, domain_label) in enumerate(tqdm(train_loader, position=1, leave=False)):
                    p = float((batch_idx + epoch * len(train_loader)) / (config.finetune2_epochs * len(train_loader)))
                    alpha = 2. / (1. + np.exp(-10 * p)) - 1
                    ls = ls.to(config.device)
                    le = le.to(config.device)
                    hi = hi.to(config.device)

                    for p in self.model.parameters():
                        p.requires_grad = False

                    self.optimizer_D.zero_grad()
                    output_real = self.discriminator(hi)
                    label_real = torch.ones(output_real.size()).to(config.device)
                    real_loss = self.criterion(output_real, label_real)
                    fake_data, _ = self.model(ls, le, alpha)
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

                    high, domain_class = self.model(ls, le, alpha)
                    output_real = self.discriminator(high)
                    self.optimizer_G.zero_grad()
                    label_real = torch.ones(output_real.size()).to(config.device)
                    real_loss = self.criterion(output_real, label_real)
                    error = (config.interval + 2) * self.criterion(high, hi) + 1e-3 * real_loss
                    generator_loss += error.mean().item()
                    if "ensemble_training" in config.tags and config.domain_backprop:
                        domain_label = F.one_hot(domain_label, num_classes=len(config.pretrain_vars)).float().to(config.device)
                        domain_loss = (config.interval + 2) * self.domain_criterion(domain_class, domain_label) / 10
                        domain_total_loss += domain_loss.mean().item()
                        error += domain_loss
                        domain_acc += torch.sum(torch.argmax(domain_class, dim=1) == torch.argmax(domain_label,dim=1)).detach().cpu().item()
                    error.backward()
                    self.optimizer_G.step()

                    for p in self.discriminator.parameters():
                        p.requires_grad = True
                domain_acc /= (len(train_loader) * config.batch_size)
                finetune2_logs["generator_loss"].append(generator_loss)
                finetune2_logs["discriminator_loss"].append(discriminator_loss)
                finetune2_logs["domain_loss"].append(domain_total_loss)
                tqdm.write(f'FT2 G loss: {generator_loss}, D loss: {discriminator_loss}')
                tqdm.write(f'FT2 domain loss: {domain_total_loss}')
                tqdm.write(f'FT2 domain accuracy: {domain_acc*100:.2f}%')
                config.log({"FT2 G loss": generator_loss, "FT2 D loss": discriminator_loss})
                config.log({"FT2 domain loss": domain_total_loss})
                config.log({"FT2 domain accuracy": domain_acc*100})
            time_end = time.time()
            finetune2_time_cost = time_end - time_start
            print('FT2 time cost', finetune2_time_cost, 's')
            finetune2_logs["time_cost"] = finetune2_time_cost
            self.save_model('finetune2', finetune2_logs)
            torch.save(self.discriminator.state_dict(), self.experiment_dir + f'/ft2_D.pth')
            torch.cuda.empty_cache()
        return pretrain_time_cost, finetune1_time_cost, finetune2_time_cost, self.model, self.discriminator

    def inference(self, load_model=None, write_to_file=True, disable_jump=False):
        assert type(self.dataset) == Dataset, "Only support single dataset inference"
        self.predict_data = []
        if write_to_file:
            Path(self.inference_dir).mkdir(parents=True, exist_ok=True)
            if not disable_jump:
                if os.path.exists(self.experiment_dir + '/inference.json'):
                    with open(self.experiment_dir + '/inference.json', 'r') as f:
                        inference_logs = json.load(f)
                        self.inference_logs = inference_logs
                        return inference_logs["PSNR"], inference_logs["PSNR_list"]
        self.model.eval()
        if load_model:
            self.load_model(self.experiment_dir + f'/{load_model}.pth')
        print('=======Inference========')
        config.log({"status": 3})
        start_time = time.time()
        lo_res_interval_loader = self.dataset.get_data("inference")
        for ind, (ls, le) in enumerate(tqdm(lo_res_interval_loader)):
            ls = ls.to(config.device)
            le = le.to(config.device)
            ls = ls.unsqueeze(1)
            le = le.unsqueeze(1)
            with torch.no_grad():
                pred, _ = self.model(ls, le)
                pred = pred.detach().cpu().numpy()
                for b in range(pred.shape[0]):
                    for j in range(0 if b + ind == 0 else 1, self.interval + 2):
                        data = pred[b][j]
                        data = np.asarray(data, dtype='<f')
                        data = data.flatten('F')
                        if write_to_file:
                            data.tofile(
                                f'{self.inference_dir}{self.dataset.dataset}-{self.dataset.selected_var}-{ind * config.batch_size * (self.interval + 1) + b * (self.interval + 1) + j + 1}.raw',
                                format='<f')
                        self.predict_data.append(data)
        end_time = time.time()
        time_cost = end_time - start_time
        print('Inference time cost', time_cost, 's')
        PSNR, PSNR_list = self.psnr()
        inference_logs = {"time_cost": time_cost, "PSNR": PSNR, "PSNR_list": PSNR_list}
        self.inference_logs = inference_logs
        if write_to_file:
            with open(self.experiment_dir + '/inference.json', 'w') as f:
                json.dump(inference_logs, f, indent=4)
        return PSNR, PSNR_list

    def psnr(self):
        print("=======Evaluating========")
        PSNR_list = []
        for ind, cmp in enumerate(tqdm(self.predict_data)):
            GT = self.dataset.hi_res[ind]
            GT = GT.flatten('F')
            GT_range = GT.max() - GT.min()
            MSE = np.mean((GT - cmp) ** 2)
            PSNR = 20 * np.log10(GT_range) - 10 * np.log10(MSE)
            PSNR_list.append(PSNR)
        print(f"PSNR is {np.mean(PSNR_list)}")
        print(f"array:\n {PSNR_list}")
        return np.mean(PSNR_list), PSNR_list

    def save_plot(self):
        if self.ensemble_iter is not None:
            desc = f'#{config.run_id}{f" E.{self.ensemble_iter}"}: {config.dataset} {self.dataset.selected_var} PSNR'
            name = f'PSNR-E.{self.ensemble_iter}'
        elif self.run_cycle is not None:
            desc = f'#{config.run_id}{f" C.{self.run_cycle}"}: {config.dataset} {self.dataset.selected_var} PSNR'
            name = f'PSNR-C.{self.run_cycle}'
        else:
            desc = f'#{config.run_id}: {config.dataset} {self.dataset.selected_var} PSNR'
            name = 'PSNR'
        x = self.inference_logs["PSNR_list"]
        plt.clf()
        axes = plt.gca()
        axes.set_ylim([0, np.max(x) + 5])
        plt.plot(x)
        plt.axhline(y=self.inference_logs["PSNR"], color='r', linestyle='--')
        plt.xlabel('Frame')
        plt.ylabel('PSNR')
        plt.title(desc)
        plt.yticks(list(plt.yticks()[0]) + [self.inference_logs["PSNR"]])
        plt.savefig(self.experiment_dir + f'/{name}.png', dpi=300)
        config.log({name: wandb.Image(self.experiment_dir + f'/{name}.png', caption=f'{desc}')})
