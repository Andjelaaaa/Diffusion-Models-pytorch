import os, random
from pathlib import Path
import pandas as pd
import re
# from kaggle import api
import torch
import imageio
import torchvision
import torchvision.transforms as T
import numpy as np
from PIL import Image
# from fastdownload import FastDownload
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from torch.utils.data import Dataset, DataLoader, Subset
import nibabel as nib
import numpy as np
from torch.nn.functional import interpolate
from monai.data import DataLoader, Dataset
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    RandSpatialCropd,
    CenterSpatialCropd,
    Resized,
    Spacingd,
    NormalizeIntensityd,
    ToTensord,
)

cifar_labels = "airplane,automobile,bird,cat,deer,dog,frog,horse,ship,truck".split(",")
alphabet_labels = "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z".split(" ")


def set_seed(s, reproducible=False):
    "Set random seed for `random`, `torch`, and `numpy` (where available)"
    try: torch.manual_seed(s)
    except NameError: pass
    try: torch.cuda.manual_seed_all(s)
    except NameError: pass
    try: np.random.seed(s%(2**32-1))
    except NameError: pass
    random.seed(s)
    if reproducible:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def untar_data(url, force_download=False, base='./datasets'):
    d = FastDownload(base=base)
    return d.get(url, force=force_download, extract_key='data')


def get_alphabet(args):
    get_kaggle_dataset("alphabet", "thomasqazwsxedc/alphabet-characters-fonts-dataset")
    train_transforms = T.Compose([
        T.Grayscale(),
        T.ToTensor(),])
    train_dataset = torchvision.datasets.ImageFolder(root="./alphabet/Images/Images/", transform=train_transforms)
    if args.slice_size>1:
        train_dataset = torch.utils.data.Subset(train_dataset, indices=range(0, len(train_dataset), args.slice_size))
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    return train_dataloader, None

def get_cifar(cifar100=False, img_size=64):
    "Download and extract CIFAR"
    cifar10_url = 'https://s3.amazonaws.com/fast-ai-sample/cifar10.tgz'
    cifar100_url = 'https://s3.amazonaws.com/fast-ai-imageclas/cifar100.tgz'
    if img_size==32:
        return untar_data(cifar100_url if cifar100 else cifar10_url)
    else:
        get_kaggle_dataset("datasets/cifar10_64", "joaopauloschuler/cifar10-64x64-resized-via-cai-super-resolution")
        return Path("datasets/cifar10_64/cifar10-64")

def get_kaggle_dataset(dataset_path, # Local path to download dataset to
                dataset_slug, # Dataset slug (ie "zillow/zecon")
                unzip=True, # Should it unzip after downloading?
                force=False # Should it overwrite or error if dataset_path exists?
               ):
    '''Downloads an existing dataset and metadata from kaggle'''
    if not force and Path(dataset_path).exists():
        return Path(dataset_path)
    api.dataset_metadata(dataset_slug, str(dataset_path))
    api.dataset_download_files(dataset_slug, str(dataset_path))
    if unzip:
        zipped_file = Path(dataset_path)/f"{dataset_slug.split('/')[-1]}.zip"
        import zipfile
        with zipfile.ZipFile(zipped_file, 'r') as zip_ref:
            zip_ref.extractall(Path(dataset_path))
        zipped_file.unlink()

def one_batch(dl):
    return next(iter(dl))
        

def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()

def plot_generated_images(x, figsize=(15, 15)):
    # Move tensor to CPU and detach from computation graph if needed
    x = x.cpu().detach()

    # Select the middle slice along the depth dimension for each image
    num_images = x.shape[0]
    middle_slice = x.shape[2] // 2  # 64 // 2 = 32, the middle slice

    # Set up the plot based on the number of images
    fig, axes = plt.subplots(1, num_images, figsize=figsize)
    axes = axes if num_images > 1 else [axes]  # Ensure axes is iterable

    # Plot the middle slice of each 3D image
    for i in range(num_images):
        # Select the middle slice along the depth dimension and remove the channel dimension
        img_slice = x[i, 0, middle_slice, :, :]  # Shape: (74, 74) for grayscale
        
        # Display the selected slice
        axes[i].imshow(img_slice, cmap="gray")
        axes[i].axis("off")  # Hide axes for a cleaner look
        axes[i].set_title(f"Image {i+1} - Depth Slice {middle_slice}")

    plt.show()

def create_animation(folder_path, output_file='animation.gif'):
    # Collect and sort PNG images in the folder
    images = sorted(
        [img for img in os.listdir(folder_path) if img.endswith('.png')],
        key=lambda x: int(re.search(r'\d+', x).group())  # Extract numeric part and sort by it
    )
    if not images:
        raise ValueError("No PNG images found in the specified folder.")

    # Load the first image to set up the plot
    fig, ax = plt.subplots()
    img = Image.open(os.path.join(folder_path, images[0]))
    im = ax.imshow(img)

    def update(frame):
        # Update the frame by loading the next image
        image_path = os.path.join(folder_path, images[frame])
        img = Image.open(image_path)
        im.set_array(img)

        # Extract epoch number from filename (assuming it's in the filename as a number)
        epoch_number = os.path.splitext(images[frame])[0]  # Remove .png to get just the filename part
        ax.set_title(f"Epoch: {epoch_number.split('-')[-1]}")

        return [im]

    # Create the animation using FuncAnimation
    
    anim = FuncAnimation(fig, update, frames=len(images), interval=len(images), blit=True)

    # Save the animation as a GIF
    anim.save(output_file, writer='pillow', fps=1000 // len(images))

    print(f"Animation saved as {output_file}")

def save_3d_images(images, path, slices=None, **kwargs):
    """
    Saves a grid of slices from 3D images.

    Args:
        images (torch.Tensor): 5D tensor of shape (B, C, D, H, W)
        path (str): Path to save the image
        slices (list or None): List of slice indices to include. If None, use middle slices.
        **kwargs: Additional keyword arguments for make_grid
    """
    # Ensure images are on CPU
    images = images.cpu()
    B, C, D, H, W = images.shape
    print('images shape', images.shape)

    if slices is None:
        # Use the middle slice if no specific slices are provided
        slices = [D // 2]

    # Collect slices
    slice_images = []
    for idx in slices:
        # Extract the slice at the specified depth
        slice_img = images[:, :, idx, :, :]  # Shape: (B, C, H, W)
        slice_images.append(slice_img)

    # Concatenate slices along the batch dimension
    slice_images = torch.cat(slice_images, dim=0)  # Shape: (B * num_slices, C, H, W)

    # Create a grid of images
    grid = torchvision.utils.make_grid(slice_images, **kwargs)

    # Convert to numpy array and save
    ndarr = grid.permute(1, 2, 0).numpy()
    im = Image.fromarray((ndarr * 255).astype(np.uint8))
    im.save(path)

def save_3d_animation(images, path):
    """
    Saves an animation (GIF) of the 3D volume slices.

    Args:
        images (torch.Tensor): 5D tensor of shape (B, C, D, H, W)
        path (str): Path to save the GIF
    """
    images = images.squeeze().cpu().numpy()  # Shape: (D, H, W)
    frames = []
    for i in range(images.shape[0]):
        frame = (images[i] * 255).astype(np.uint8)
        frames.append(frame)
    # Save as GIF
    imageio.mimsave(path, frames, fps=10)

def save_nifti(tensor, filename):
    tensor = tensor.squeeze().cpu().numpy()  # Remove batch and channel dimensions
    nifti_img = nib.Nifti1Image(tensor, affine=np.eye(4))
    nib.save(nifti_img, filename)

def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)

# Custom Dataset Class
class NiftiDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the data (train_folder or val_folder).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = self._gather_samples()

    def _gather_samples(self):
        samples = []
        for sub_dir in os.listdir(self.root_dir):
            sub_path = os.path.join(self.root_dir, sub_dir)
            if os.path.isdir(sub_path):
                for trio_dir in os.listdir(sub_path):
                    trio_path = os.path.join(sub_path, trio_dir)
                    if os.path.isdir(trio_path):
                        # Collect all NIfTI files in this trio directory
                        for filename in os.listdir(trio_path):
                            if filename.endswith('.nii.gz'):
                                file_path = os.path.join(trio_path, filename)
                                samples.append(file_path)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        # Load the NIfTI image
        img = nib.load(img_path).get_fdata()
        # Convert to a torch tensor
        img = torch.from_numpy(img).float()
        # Add a channel dimension if needed (assuming grayscale images)
        if img.dim() == 3:
            img = img.unsqueeze(0)  # Shape: (1, D, H, W)
        # Apply transforms if any
        if self.transform:
            for t in self.transform:
                img = t(img)
        return img

def get_transforms(is_train=True, args=None):
    if is_train:
        transforms = Compose([
            LoadImaged(keys=['image']),
            EnsureChannelFirstd(keys=['image']),
            # RandSpatialCropd(keys=['image'], roi_size=(args.crop_depth, args.crop_height, args.crop_width), random_center=True, random_size=False),
            Spacingd(keys=['image'], pixdim=(args.pixdim, args.pixdim, args.pixdim), mode=('bilinear')),
            # Resized(keys=['image'], spatial_size=(args.depth_size, args.img_size, args.img_size)),
            NormalizeIntensityd(keys=['image'], nonzero=True),#, channel_wise=True),
            ToTensord(keys=['image']),
        ])
    else:
        transforms = Compose([
            LoadImaged(keys=['image']),
            EnsureChannelFirstd(keys=['image']),
            # CenterSpatialCropd(keys=['image'], roi_size=(args.crop_depth, args.crop_height, args.crop_width)),
            # Resized(keys=['image'], spatial_size=(args.img_depth, args.img_size, args.img_size)),
            Spacingd(keys=['image'], pixdim=(3.0, 3.0, 3.0), mode=('bilinear')),
            NormalizeIntensityd(keys=['image'], nonzero=True, channel_wise=True),
            ToTensord(keys=['image']),
        ])
    return transforms

# def get_data_dicts(root_dir):
#     data_dicts = []
#     for sub_dir in os.listdir(root_dir):
#         sub_path = os.path.join(root_dir, sub_dir)
#         if os.path.isdir(sub_path):
#             for trio_dir in os.listdir(sub_path):
#                 trio_path = os.path.join(sub_path, trio_dir)
#                 if os.path.isdir(trio_path):
#                     for filename in os.listdir(trio_path):
#                         if filename.endswith('.nii.gz'):
#                             file_path = os.path.join(trio_path, filename)
#                             data_dicts.append({'image': file_path})
#     return data_dicts
def get_data_dict(filepath, mode='train'):
    # Load the data from the CSV file
    df = pd.read_csv(filepath)

    data_dict = []
    
    # Filter the DataFrame based on the mode in the 'traintest' column
    if mode == 'train':
        filtered_df = df[df['traintest'] == 'train']
    elif mode == 'test':
        filtered_df = df[df['traintest'] == 'test']
    else:
        raise ValueError("Mode must be 'train' or 'test'")
    
    for i in range(len(filtered_df)):
        # Use double quotes for column access inside an f-string
        data_dict.append({'image': f"/home/andjela/joplin-intra-inter/CP_rigid_trios/CP/{filtered_df['sub_id_bids'].iloc[i]}/{filtered_df['trio_id'].iloc[i]}/{filtered_df['scan_id'].iloc[i]}.nii.gz"})
    
    return data_dict
# Custom Transforms for 3D Data
class Resize3D:
    def __init__(self, size):
        self.size = size  # size should be a tuple (D, H, W)

    def __call__(self, img):
        # img is a torch tensor with shape (C, D, H, W)
        img = interpolate(img.unsqueeze(0), size=self.size, mode='trilinear', align_corners=False)
        return img.squeeze(0)

class Normalize3D:
    def __call__(self, img):
        print('img max-min', img.max(), img.min())
        mean = img.mean()
        std = img.std()
        # To avoid division by zero
        if std == 0:
            std = 1
        return (img - mean) / std

def get_data(args):
    # Prepare data dictionaries
    train_data_dicts = get_data_dict(args.dataset_path)
    val_data_dicts = get_data_dict(args.dataset_path, mode='test')

    # Define transforms
    train_transforms = get_transforms(is_train=True, args=args)
    val_transforms = get_transforms(is_train=False, args=args)

    # Create datasets
    train_dataset = Dataset(data=train_data_dicts, transform=train_transforms)
    val_dataset = Dataset(data=val_data_dicts, transform=val_transforms)

    # Create data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return train_dataloader, val_dataloader
# def get_data(args):
#     # Define the desired output size for your 3D images
#     desired_size = (args.depth_size, args.img_size, args.img_size)  # For example, (64, 128, 128)

#     # Define the transforms as a list
#     train_transforms = [Resize3D(desired_size), Normalize3D()]
#     val_transforms = [Resize3D(desired_size), Normalize3D()]

#     # Create datasets
#     train_dataset = NiftiDataset(os.path.join(args.dataset_path, 'train_folder'), transform=train_transforms)
#     val_dataset = NiftiDataset(os.path.join(args.dataset_path, 'val_folder'), transform=val_transforms)
    
#     if args.slice_size > 1:
#         indices = range(0, len(train_dataset), args.slice_size)
#         train_dataset = Subset(train_dataset, indices)
#         indices = range(0, len(val_dataset), args.slice_size)
#         val_dataset = Subset(val_dataset, indices)

#     train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
#     val_dataloader = DataLoader(val_dataset, batch_size=2 * args.batch_size, shuffle=False, num_workers=args.num_workers)
#     return train_dataloader, val_dataloader


# def get_data(args):
#     train_transforms = torchvision.transforms.Compose([
#         T.Resize(args.img_size + int(.25*args.img_size)),  # args.img_size + 1/4 *args.img_size
#         T.RandomResizedCrop(args.img_size, scale=(0.8, 1.0)),
#         T.ToTensor(),
#         T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#     ])

#     val_transforms = torchvision.transforms.Compose([
#         T.Resize(args.img_size),
#         T.ToTensor(),
#         T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#     ])

#     train_dataset = torchvision.datasets.ImageFolder(os.path.join(args.dataset_path, 'train_folder'), transform=train_transforms)
#     val_dataset = torchvision.datasets.ImageFolder(os.path.join(args.dataset_path, 'val_folder'), transform=val_transforms)
    
#     if args.slice_size>1:
#         train_dataset = torch.utils.data.Subset(train_dataset, indices=range(0, len(train_dataset), args.slice_size))
#         val_dataset = torch.utils.data.Subset(val_dataset, indices=range(0, len(val_dataset), args.slice_size))

#     train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
#     val_dataset = DataLoader(val_dataset, batch_size=2*args.batch_size, shuffle=False, num_workers=args.num_workers)
#     return train_dataloader, val_dataset

def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)

def mk_folders(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)
