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
    def __init__(
        self, 
        noise_steps=1000, 
        beta_start=1e-4, 
        beta_end=0.02, 
        img_size=(78, 78, 64),  # (D, H, W)
        device="cuda"
    ):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size  # (D, H, W)
        self.device = device

        # Prepare the noise schedule
        self.beta = self.prepare_noise_schedule().to(self.device)  # Shape: (noise_steps,)
        self.alpha = 1.0 - self.beta  # Shape: (noise_steps,)
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)  # Shape: (noise_steps,)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        """
        Adds noise to the images x at timestep t.
        
        Args:
            x (torch.Tensor): Original images of shape (B, C, D, H, W)
            t (torch.Tensor): Timesteps of shape (B,)
        
        Returns:
            x_t (torch.Tensor): Noisy images at timestep t
            ε (torch.Tensor): The noise added to the images
        """
        # Ensure that t is of shape (B,)
        if t.dim() == 0:
            t = t.unsqueeze(0)

        # Reshape alpha_hat[t] to (B, 1, 1, 1, 1) to broadcast correctly
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None, None]  # Shape: (B, 1, 1, 1, 1)
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None, None]

        ε = torch.randn_like(x)  # Noise of same shape as x
        x_t = sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * ε  # Add noise to the images
        return x_t, ε

    def sample_timesteps(self, n):
        """
        Samples random timesteps for a batch of size n.
        
        Args:
            n (int): Batch size
        
        Returns:
            t (torch.Tensor): Random timesteps of shape (n,)
        """
        return torch.randint(low=0, high=self.noise_steps, size=(n,))

    def sample(self, model, n):
        """
        Generates n new images by reverse diffusion.

        Args:
            model: The trained diffusion model
            n (int): Number of images to sample

        Returns:
            x (torch.Tensor): Generated images of shape (n, C, D, H, W)
        """
        logging.info(f"Sampling {n} new images...")
        model.eval()
        c_in = 1  # Number of input channels (adjust if necessary)
        D, H, W = self.img_size  # Unpack image dimensions

        with torch.no_grad():
            x = torch.randn((n, c_in, D, H, W)).to(self.device)  # Start from random noise
            for i in tqdm(reversed(range(self.noise_steps)), position=0, desc='Sampling'):
                t = torch.full((n,), i, dtype=torch.long).to(self.device)  # Timesteps t of shape (n,)

                # Get model prediction of noise at timestep t
                predicted_noise = model(x, t)  # Should output shape (n, c_in, D, H, W)

                # Get alpha, alpha_hat, and beta at timestep t
                alpha = self.alpha[t][:, None, None, None, None]  # Shape: (n, 1, 1, 1, 1)
                alpha_hat = self.alpha_hat[t][:, None, None, None, None]
                beta = self.beta[t][:, None, None, None, None]

                # If not the last step, add noise
                if i > 0:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)

                # Update x using the reverse diffusion formula
                x = (1 / torch.sqrt(alpha)) * (
                    x - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * predicted_noise
                ) + torch.sqrt(beta) * noise

            # After the loop, x should be the generated images
        model.train()
        # Normalize x to [0, 1] range
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)  # Convert to uint8 if necessary
        return x

def plot_input(train_dataloader):
    train_img = next(iter(train_dataloader))
    print(train_img.keys())
    print('SIZE', train_img['image'][0].size())
    image_path = train_img['image'][0]
    # Assuming train_img['image'] is a batch of images
    image = train_img['image'][0][0]  # Shape: (D, H, W)

    # Slice indices to visualize
    slice_indices = [int(image.shape[2]/3), int(image.shape[2]/2.5), int(image.shape[2]/2), int(image.shape[2]/1.5)]
    # Create a figure with 2x2 subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    # Flatten axs for easy iteration
    axs = axs.flatten()

    for i, idx in enumerate(slice_indices):
        img_slice = image[:, :, idx]  # Get the slice at depth index `idx`
        axs[i].imshow(img_slice, cmap='gray')
        axs[i].set_title(f'Slice at index {idx}')
        axs[i].axis('off')  # Hide axis ticks and labels
    # Add the image path as a suptitle
    # fig.suptitle(f'Image Path: {image_path}', fontsize=12)
    plt.tight_layout()
    plt.show()

def train(args):
    setup_logging(args.run_name)
    device = args.device
    train_dataloader, val_dataloader = get_data(args)
    model = UNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=(args.img_depth, args.img_size, args.img_size), device=device)
    l = len(train_dataloader)
    print('len(train_dataloader):', l)

    # To visualize input images
    # plot_input(train_dataloader)
    # return True
    
    
    # x = torch.randn(batch_size, channels, depth, height, width)
    # t = torch.randint(0, 1000, (batch_size,))

    # model = UNet(c_in=channels, c_out=channels)
    # output = model(x, t)
    # print(output.shape)

    # Initialize W&B and global_step
    wandb.init(project="DiffusionModel", name=args.run_name, config=vars(args))
    wandb.watch(model, log="all")
    global_step = 0  # Initialize global step

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(train_dataloader)
        for i, batch_data in enumerate(pbar):
            images = batch_data['image'].to(device)
            # print('images shape before:', images.shape)
            # Permute to (B, C, D, H, W)
            images = images.permute(0, 1, 4, 2, 3)
            # print('images shape after:', images.shape)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            predicted_noise = model(x_t, t)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(MSE=loss.item())
            # Log MSE to W&B
            wandb.log({"MSE": loss.item()}, step=global_step)
            # batch_step = epoch * l + i
            global_step += 1  # Increment global step after each batch

        # After each epoch, sample images
        sampled_images = diffusion.sample(model, n=1)  # Shape: [1, 1, D, H, W]

        # Normalize images
        sampled_images = sampled_images.float() / 255.0

        # Save grid of slices
        save_path = os.path.join("results", args.run_name, f"epoch-{epoch}.png")
        gif_path = os.path.join("results", args.run_name, f"epoch-{epoch}.gif")
        slices = [int(args.img_depth/3), int(args.img_depth/2.5), int(args.img_depth/2), int(args.img_depth/1.5)]  # Specify slices
        print('Slice indices:', slices)
        save_3d_images(sampled_images, save_path, slices=slices, nrow=len(slices), padding=2)

        # **Create the animation**
        # save_3d_animation(sampled_images, gif_path)

        # Log to W&B using the current global step
        wandb.log({"Sampled Images": wandb.Image(save_path)}, step=global_step)
        # wandb.log({"Sampled Images Animation": wandb.Video(gif_path)}, step=global_step)

        # Save the NIfTI file
        # nifti_path = os.path.join("results", args.run_name, f"epoch-{epoch}_sample-{i}.nii.gz")
        # save_nifti(sampled_images, nifti_path)

        global_step += 1  # Increment after logging images

        # Save the model checkpoint
        torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))

    # Finish W&B run
    wandb.finish()



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
    args.run_name = "DDPM_Unconditional_it2"
    # batch_size = 3
    # channels = 3
    # depth = 116
    # height = 116
    # width = 95
    args.epochs = 300
    args.batch_size = 1
    args.pixdim = 3.0
    args.img_size = 78
    args.img_depth = 64
    # No subsampling
    args.slice_size = 1
    # args.crop_depth = 60   # Desired crop size along depth
    # args.crop_height = 120  # Desired crop size along height
    # args.crop_width = 120   # Desired crop size along width
    args.num_workers = 5
    # args.dataset_path = "/home/andjela/Documents/longitudinal-pediatric-completion/data/CP/"
    args.dataset_path = "/home/andjela/joplin-intra-inter/CP_rigid_trios/CP/trios_sorted_by_age.csv"
    args.device = "cuda"
    args.lr = 3e-5
    train(args)


if __name__ == '__main__':
    # launch()
    # create_animation('results/DDPM_Unconditional_it2/', duration=214)
    device = "cuda"
    model = UNet().to(device)
    ckpt = torch.load("models/DDPM_Unconditional_it2/ckpt.pt")
    model.load_state_dict(ckpt)
    diffusion = Diffusion(img_size=(64, 74, 74), device=device)
    x = diffusion.sample(model, 4)
    plot_generated_images(x)
    # plt.figure(figsize=(32, 32))
    # plt.imshow(torch.cat([
    #     torch.cat([i for i in x.cpu()], dim=-1),
    # ], dim=-2).permute(1, 2, 0).cpu())
    # plt.show()
