import os
import torch
import torch.nn as nn
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('TkAgg')
from tqdm import tqdm
from torch import optim
from utils import *
from modules import UNet
import logging
import wandb
from torchvision.utils import make_grid

# from torch.utils.tensorboard import SummaryWriter

# logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x


def train(args):
    setup_logging(args.run_name)
    device = args.device
    train_dataloader, val_dataloader = get_data(args)
    model = UNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.img_size, device=device)
    l = len(train_dataloader)
    print('len(train_dataloader):', l)

    train_img = next(iter(train_dataloader))
    print(train_img.keys())
    print('SIZe', train_img['image'][0].size())
    # Assuming train_img['image'] is a batch of images
    image = train_img['image'][0][0]  # Shape: (D, H, W)

    # Indices of slices to plot
    # slice_indices = [10, 20, 32, 50]
    slice_indices = [100, 105, 32, 50]

    # Create a figure with 2x2 subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    # Flatten axs for easy iteration
    axs = axs.flatten()

    for i, idx in enumerate(slice_indices):
        img_slice = image[:, :, idx]  # Get the slice at depth index `idx`
        axs[i].imshow(img_slice, cmap='gray')
        axs[i].set_title(f'Slice at index {idx}')
        axs[i].axis('off')  # Hide axis ticks and labels

    plt.tight_layout()
    plt.show()

    # Initialize W&B
    # wandb.init(project="DiffusionModel", name=args.run_name, config=vars(args))
    # wandb.watch(model, log="all")

    # for epoch in range(args.epochs):
    #     logging.info(f"Starting epoch {epoch}:")
    #     pbar = tqdm(train_dataloader)
    #     for i, batch_data in enumerate(pbar):
    #         images = batch_data['image'].to(device)
    #         images = images.to(device)
    #         t = diffusion.sample_timesteps(images.shape[0]).to(device)
    #         x_t, noise = diffusion.noise_images(images, t)
    #         predicted_noise = model(x_t, t)
    #         loss = mse(noise, predicted_noise)

    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()

    #         pbar.set_postfix(MSE=loss.item())
    #         # Log MSE to W&B
    #         wandb.log({"MSE": loss.item()}, step=epoch * l + i)

    #     sampled_images = diffusion.sample(model, n=images.shape[0])
    #     save_images(sampled_images, os.path.join("results", args.run_name, f"epoch-{epoch}.jpg"))

    #     # Prepare images for logging
    #     sampled_images = sampled_images.float() / 255.0  # Normalize to [0, 1]
    #     # Log images to W&B
    #     wandb.log({"Sampled Images": [wandb.Image(img) for img in sampled_images]}, step=epoch)

    #     torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))

    # # Finish W&B run
    # wandb.finish()



def launch():
    # import argparse
    # parser = argparse.ArgumentParser()
    # args = parser.parse_args()
    # args.run_name = "DDPM_Unconditional"
    # args.epochs = 500
    # args.batch_size = 12
    # args.image_size = 64
    # args.dataset_path = r"C:\Users\dome\datasets\landscape_img_folder"
    # args.device = "cuda"
    # args.lr = 3e-4
    # train(args)
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "DDPM_Unconditional"
    args.epochs = 10
    args.batch_size = 3
    args.img_size = 128
    args.depth_size = 64
    # No subsampling
    args.slice_size = 1
    args.crop_depth = 60   # Desired crop size along depth
    args.crop_height = 120  # Desired crop size along height
    args.crop_width = 120   # Desired crop size along width
    args.num_workers = 5
    args.dataset_path = "/home/andjela/Documents/longitudinal-pediatric-completion/data/CP/"
    args.device = "cuda"
    args.lr = 3e-4
    train(args)


if __name__ == '__main__':
    launch()
    device = "cuda"
    model = UNet().to(device)
    # ckpt = torch.load("./working/orig/ckpt.pt")
    # model.load_state_dict(ckpt)
    diffusion = Diffusion(img_size=512, device=device)
    # x = diffusion.sample(model, 8)
    # print(x.shape)
    # plt.figure(figsize=(32, 32))
    # plt.imshow(torch.cat([
    #     torch.cat([i for i in x.cpu()], dim=-1),
    # ], dim=-2).permute(1, 2, 0).cpu())
    # plt.show()
