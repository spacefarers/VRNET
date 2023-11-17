from tqdm import tqdm
from torch import nn, optim
from dataset_io import Dataset
import torch.nn.functional as F
import torch
import os
import numpy as np
from pathlib import Path
from model import weights_init_kaiming


class Trainer:
    def __init__(self, model, discriminator, dataset: Dataset, run_id, experiments_dir):
        self.model = nn.DataParallel(model)
        self.discriminator = nn.DataParallel(discriminator)
        self.model.cuda()
        self.discriminator.cuda()
        self.model.apply(weights_init_kaiming)
        self.discriminator.apply(weights_init_kaiming)
        self.dataset = dataset
        self.optimizer_G = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999))
        self.optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0004, betas=(0.9, 0.999))
        self.criterion = nn.MSELoss()
        self.run_id = f"{run_id:03d}"
        self.stages = ['pretrain', 'finetune1', 'finetune2']
        self.experiments_dir = os.path.join(experiments_dir, self.run_id)
        self.inference_dir = self.experiments_dir + "/inference/"

    def load_model(self, stage):
        self.model.load_state_dict(torch.load(self.experiments_dir + f'/{stage}.pth'))

    def jump_to_progress(self):
        for stage in reversed(self.stages):
            if os.path.exists(self.experiments_dir + f'/{stage}.pth'):
                print('Jump to %s' % stage)
                self.load_model(stage)
                return stage
        return None

    def save_model(self, stage, logs=None):
        Path(self.experiments_dir).mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), self.experiments_dir + f'/{stage}.pth')
        if logs:
            with open(self.experiments_dir + f'/{stage}_logs.txt', 'w') as f:
                f.write('\n'.join(logs))

    def train(self, pretrain_epochs, finetune1_epochs, finetune2_epochs, interval, crop_times):
        stage = self.jump_to_progress()

        if stage is None and pretrain_epochs > 0:
            # Pretrain
            print('=======Pretrain========')
            pretrain_logs = []
            for _ in tqdm(range(pretrain_epochs)):
                loss_mse = 0
                train_loader = self.dataset.spacial_dataloader(interval, crop_times)
                for batch_idx, (ls, le, li) in tqdm(enumerate(train_loader)):
                    ls = ls.cuda()
                    le = le.cuda()
                    li = li.cuda()
                    high = self.model(ls, le)
                    self.optimizer_G.zero_grad()
                    error = (interval + 2) * self.criterion(F.interpolate(high.permute(1, 0, 2, 3, 4),
                                                                          size=[self.dataset.crop_size[0],
                                                                                self.dataset.crop_size[1],
                                                                                self.dataset.crop_size[2]],
                                                                          mode='trilinear'),
                                                            li)
                    error.backward()
                    loss_mse += error.mean().item()
                    self.optimizer_G.step()
                pretrain_logs.append(str(loss_mse))
                tqdm.write('Pretrain loss: %f' % loss_mse)
            self.save_model('pretrain', pretrain_logs)
            torch.cuda.empty_cache()
        if (stage is None or stage == 'pretrain') and finetune1_epochs > 0:
            # Finetune 1
            print('=======Finetune 1========')
            finetune1_logs = []
            for _ in tqdm(range(1, finetune1_epochs + 1)):
                loss_mse = 0
                train_loader = self.dataset.spacial_temporal_dataloader(interval, crop_times)
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
                finetune1_logs.append(str(loss_mse))
                print('Finetune 1 loss: %f' % loss_mse)
            self.save_model('finetune1', finetune1_logs)
            torch.cuda.empty_cache()

        if stage != 'finetune2' and finetune2_epochs > 0:
            # Finetune 2
            tqdm.write('=======Finetune 2========')
            finetune2_logs = []
            for _ in tqdm(range(1, finetune2_epochs + 1)):
                generator_loss = 0
                discriminator_loss = 0
                train_loader = self.dataset.spacial_temporal_dataloader(interval, crop_times)
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
                finetune2_logs.append(f'{generator_loss} {discriminator_loss}')
                tqdm.write(f'Generator loss: {generator_loss} discriminator loss: {discriminator_loss}')
            self.save_model('finetune2', finetune2_logs)
            torch.cuda.empty_cache()

    def inference(self, interval,load_model=None):
        if os.listdir(self.inference_dir):
            print("Inference Already Done")
            return
        if load_model:
            self.load_model(load_model)
        print('=======Inference========')
        Path(self.inference_dir).mkdir(parents=True, exist_ok=True)
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
                    for j in range(0, interval + 2):
                        data = s[0][j]
                        data = np.asarray(data, dtype='<f')
                        data = data.flatten('F')
                        # normalize to [-1,1]
                        data = 2 * (data - np.min(data)) / (np.max(data) - np.min(data)) - 1
                        data.tofile(
                            f'{self.inference_dir}{self.dataset.dataset}-{self.dataset.selected_var}-{i + j + 1}.raw',
                            format='<f')
