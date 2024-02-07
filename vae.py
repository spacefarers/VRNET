# Variational Autoencoder for 3D
import model
from torch import nn
import torch
import config
import numpy as np
from dataset_io import Dataset
from fire import Fire
import os
from pathlib import Path
from tqdm import tqdm


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # Encoder
        self.enc_conv1 = model.RDB(4, 16)
        self.enc_conv2 = model.RDB(16, 32)
        self.enc_fc1 = nn.Linear(32 * np.prod(config.crop_size) * config.scale ** 3, 512)
        self.enc_fc2_mean = nn.Linear(512, 256)
        self.enc_fc2_logvar = nn.Linear(512, 256)

        # Decoder
        self.dec_fc1 = nn.Linear(256, 512)
        self.dec_fc2 = nn.Linear(512, 32 * np.prod(config.crop_size) * config.scale ** 3)
        self.dec_conv1 = model.RDB(32, 16)
        self.dec_conv2 = model.RDB(16, 4)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        x = self.relu(self.enc_conv1(x))
        x = self.relu(self.enc_conv2(x))
        x = torch.flatten(x, start_dim=1)
        x = self.relu(self.enc_fc1(x))
        return self.enc_fc2_mean(x), self.enc_fc2_logvar(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        x = self.relu(self.dec_fc1(z))
        x = self.relu(self.dec_fc2(x))
        x = x.view(-1, 32, config.crop_size[0] * config.scale, config.crop_size[1] * config.scale, config.crop_size[2] * config.scale)
        x = self.relu(self.dec_conv1(x))
        x = self.sigmoid(self.dec_conv2(x))
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def main():
    total_epochs = 20
    run_id = 40
    config.tags.append("VAE")
    config.run_id = run_id
    run_id = f"{config.run_id:03d}"
    experiment_dir = os.path.join(config.experiments_dir, run_id)
    inference_dir = experiment_dir + "/inference/"
    Path(experiment_dir).mkdir(parents=True, exist_ok=True)
    M = model.prep_model(VAE())
    optimizer = torch.optim.Adam(M.parameters(), lr=5e-5, betas=(0.9, 0.999))
    dataset = Dataset(config.source_dataset, config.source_var, "all")
    criterion = nn.MSELoss()
    for epoch in range(total_epochs):
        for (low_res_window, high_res_window) in tqdm(dataset.get_augmented_data(), desc=f"Training Epoch {epoch+1}/{total_epochs}", leave=False):
            high_res_window = high_res_window.to(config.device)
            optimizer.zero_grad()
            pred, mu, logvar = M(high_res_window)
            reconstruct_loss = criterion(pred, high_res_window)
            kl_divergence = 1 + logvar - torch.square(mu) - torch.exp(logvar)
            kl_divergence = -0.5 * torch.sum(kl_divergence, dim=-1)
            loss = reconstruct_loss + kl_divergence
            loss.backward()
            optimizer.step()
            config.track({"reconstruct_loss": reconstruct_loss, "kl_divergence": kl_divergence})
        config.log_all()
        torch.save(M.state_dict(), f"{config.experiments_dir}/source_trained.pth")

def render(latent_space:torch.Tensor, M):
    M.eval()
    latent_space = latent_space.to(config.device)
    with torch.no_grad():
        pred = M.module.decode(latent_space)
        pred = pred.detach().cpu().numpy()
        for i in range(config.interval+2):
            data = pred[0][i]
            data = np.asarray(data, dtype='<f')
            data = data.flatten('F')
            data.tofile(f'{config.experiments_dir}/rendered_{i}.raw', format='<f')


def ground_truth(dataset:Dataset):
    pass

def get_random_renders():
    M = model.prep_model(VAE())
    M,_ = model.load_model(M, torch.load(f"{config.experiments_dir}/source_trained.pth"))
    render(torch.randn(1, 2048), M)


if __name__ == "__main__":
    # get_random_renders()
    main()