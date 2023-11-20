from tqdm import tqdm
from torch import nn, optim
from dataset_io import Dataset
import torch
import os
import numpy as np
from pathlib import Path
import json
import time


class Trainer:
    def __init__(self, dataset: Dataset, run_id, experiments_dir, model, discriminator=None):
        self.model = model
        self.discriminator = discriminator
        self.dataset = dataset
        self.optimizer_G = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999))
        self.optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0004, betas=(0.9, 0.999))
        self.criterion = nn.MSELoss()
        self.run_id = f"{run_id:03d}"
        self.stages = ['finetune1', 'finetune2']
        self.experiment_dir = os.path.join(experiments_dir, self.run_id)
        self.inference_dir = self.experiment_dir + "/inference/"
        self.predict_data = []

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
        Path(self.experiment_dir).mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), self.experiment_dir + f'/{stage}.pth')
        if logs:
            with open(self.experiment_dir + f'/{stage}_logs.json', 'w') as f:
                json.dump(logs, f)

    def train(self, finetune1_epochs, finetune2_epochs, interval, crop_times, disable_jump=False):
        pretrain_time_cost = 0
        finetune1_time_cost = 0
        finetune2_time_cost = 0
        stage = self.jump_to_progress() if not disable_jump else None
        if (stage is None or stage == 'pretrain') and finetune1_epochs > 0:
            # Finetune 1
            print('=======Finetune 1========')
            time_start = time.time()
            finetune1_logs = {"loss": []}
            train_loader = self.dataset.spacial_temporal_dataloader(crop_times)
            for epoch in tqdm(range(1, finetune1_epochs + 1)):
                loss_mse = 0
                for batch_idx, (ls, le, hi) in enumerate(train_loader):
                    ls = ls.cuda()
                    le = le.cuda()
                    hi = hi.cuda()
                    high = self.model(ls, le)
                    self.optimizer_G.zero_grad()
                    error = (interval + 2) * self.criterion(high, hi)
                    error.backward()
                    loss_mse += error.mean().item()
                    self.optimizer_G.step()
                finetune1_logs["loss"].append(loss_mse)
                tqdm.write(f"FT1 {epoch} loss: {loss_mse}")
            time_end = time.time()
            finetune1_time_cost = time_end - time_start
            print('FT1 time cost', finetune1_time_cost, 's')
            finetune1_logs["time_cost"] = finetune1_time_cost
            self.save_model('finetune1', finetune1_logs)
            torch.cuda.empty_cache()

        if stage != 'finetune2' and finetune2_epochs > 0:
            # Finetune 2
            tqdm.write('=======Finetune 2========')
            time_start = time.time()
            finetune2_logs = {"generator_loss": [], "discriminator_loss": []}
            train_loader = self.dataset.spacial_temporal_dataloader(crop_times)
            for epoch in tqdm(range(1, finetune2_epochs + 1)):
                generator_loss = 0
                discriminator_loss = 0
                for batch_idx, (ls, le, hi) in enumerate(train_loader):
                    ls = ls.cuda()
                    le = le.cuda()
                    hi = hi.cuda()

                    for p in self.model.parameters():
                        p.requires_grad = False

                    self.optimizer_D.zero_grad()
                    output_real = self.discriminator(hi)
                    label_real = torch.ones(output_real.size()).cuda()
                    real_loss = self.criterion(output_real, label_real)
                    fake_data = self.model(ls, le)
                    label_fake = torch.zeros(output_real.size()).cuda()
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

                    high = self.model(ls, le)
                    output_real = self.discriminator(high)
                    self.optimizer_G.zero_grad()
                    label_real = torch.ones(output_real.size()).cuda()
                    real_loss = self.criterion(output_real, label_real)
                    error = (interval + 2) * self.criterion(high, hi) + 1e-3 * real_loss
                    error.backward()
                    generator_loss += error.mean().item()
                    self.optimizer_G.step()

                    for p in self.discriminator.parameters():
                        p.requires_grad = True
                finetune2_logs["generator_loss"].append(generator_loss)
                finetune2_logs["discriminator_loss"].append(discriminator_loss)
                tqdm.write(f'FT2 {epoch} Generator loss: {generator_loss} Discriminator loss: {discriminator_loss}')
            time_end = time.time()
            finetune2_time_cost = time_end - time_start
            print('FT2 time cost', finetune2_time_cost, 's')
            finetune2_logs["time_cost"] = finetune2_time_cost
            self.save_model('finetune2', finetune2_logs)
            torch.save(self.discriminator.state_dict(), self.experiment_dir + f'/ft2_D.pth')
            torch.cuda.empty_cache()
        return pretrain_time_cost, finetune1_time_cost, finetune2_time_cost, self.model, self.discriminator

    def inference(self, interval, load_model=None):
        Path(self.inference_dir).mkdir(parents=True, exist_ok=True)
        if os.path.exists(self.experiment_dir + '/inference.json'):
            with open(self.experiment_dir + '/inference.json', 'r') as f:
                inference_logs = json.load(f)
                return inference_logs["PSNR"], inference_logs["PSNR_list"]
        if load_model:
            self.load_model(self.experiment_dir + f'/{load_model}.pth')
        print('=======Inference========')
        start_time = time.time()
        low, high = self.dataset.low_res, self.dataset.hi_res
        for i in tqdm(range(0, len(low), interval + 1)):
            if i + 1 + interval < len(low):
                ls = torch.FloatTensor(
                    low[i].reshape(1, 1, self.dataset.dims[0] // self.dataset.scale,
                                   self.dataset.dims[1] // self.dataset.scale,
                                   self.dataset.dims[2] // self.dataset.scale))
                le = torch.FloatTensor(
                    low[interval + i + 1].reshape(1, 1, self.dataset.dims[0] // self.dataset.scale,
                                                  self.dataset.dims[1] // self.dataset.scale,
                                                  self.dataset.dims[2] // self.dataset.scale))
                ls = ls.cuda()
                le = le.cuda()
                with torch.no_grad():
                    s = self.model(ls, le)
                    s = s.detach().cpu().numpy()
                    for j in range(0 if i == 0 else 1, interval + 2):
                        data = s[0][j]
                        data = np.asarray(data, dtype='<f')
                        data = data.flatten('F')
                        data.tofile(
                            f'{self.inference_dir}{self.dataset.dataset}-{self.dataset.selected_var}-{i + j + 1}.raw',
                            format='<f')
                        self.predict_data.append(data)
        end_time = time.time()
        time_cost = end_time - start_time
        print('Inference time cost', time_cost, 's')
        PSNR, PSNR_list = self.psnr()
        inference_logs = {"time_cost": time_cost, "PSNR": PSNR, "PSNR_list": PSNR_list}
        with open(self.experiment_dir + '/inference.json', 'w') as f:
            json.dump(inference_logs, f)
        return PSNR, PSNR_list

    def psnr(self):
        print("=======Evaluating========")
        PSNR_list = []
        for ind, GT in enumerate(tqdm(self.dataset.hi_res)):
            cmp = self.predict_data[ind]
            GT = GT.flatten('F')
            GT_range = GT.max() - GT.min()
            MSE = np.mean((GT - cmp) ** 2)
            PSNR = 20 * np.log10(GT_range) - 10 * np.log10(MSE)
            PSNR_list.append(PSNR)
        print(f"PSNR is {np.mean(PSNR_list)}")
        print(f"array:\n {PSNR_list}")
        return np.mean(PSNR_list), PSNR_list

    def increase_framerate(self, interval, load_model=None):
        Path(self.inference_dir).mkdir(parents=True, exist_ok=True)
        if load_model:
            self.load_model(self.experiment_dir + f'/{load_model}.pth')
        print('=======Inference========')
        low, high = self.dataset.low_res, self.dataset.hi_res
        for i in tqdm(range(len(low) - 1)):
            ls = torch.FloatTensor(
                low[i].reshape(1, 1, self.dataset.dims[0] // self.dataset.scale,
                               self.dataset.dims[1] // self.dataset.scale,
                               self.dataset.dims[2] // self.dataset.scale))
            le = torch.FloatTensor(
                low[i + 1].reshape(1, 1, self.dataset.dims[0] // self.dataset.scale,
                                   self.dataset.dims[1] // self.dataset.scale,
                                   self.dataset.dims[2] // self.dataset.scale))
            ls = ls.cuda()
            le = le.cuda()
            with torch.no_grad():
                s = self.model(ls, le)
                s = s.detach().cpu().numpy()
                for j in range(0, interval + 2):
                    data = s[0][j]
                    data = np.asarray(data, dtype='<f')
                    data = data.flatten('F')
                    data.tofile(
                        f'{self.inference_dir}{self.dataset.dataset}-{self.dataset.selected_var}-{i * 2 + j + 1}.raw',
                        format='<f')
