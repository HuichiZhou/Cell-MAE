import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import pandas as pd
import torch.nn.functional as F
from torchvision.transforms.functional import resize

class CustomDataset(Dataset):
    def __init__(self):
        # Define the mapping of folders to labels (51 folders, labels from 0 to 50)
        self.class_map = {
            'HEPG2-01': 0, 'HEPG2-02': 1, 'HEPG2-03': 2, 'HEPG2-04': 3, 'HEPG2-05': 4, 'HEPG2-06': 5, 'HEPG2-07': 6,
            'HEPG2-08': 7, 'HEPG2-09': 8, 'HEPG2-10': 9, 'HEPG2-11': 10, 'HUVEC-01': 11, 'HUVEC-02': 12, 'HUVEC-03': 13,
            'HUVEC-04': 14, 'HUVEC-05': 15, 'HUVEC-06': 16, 'HUVEC-07': 17, 'HUVEC-08': 18, 'HUVEC-09': 19, 'HUVEC-10': 20,
            'HUVEC-11': 21, 'HUVEC-12': 22, 'HUVEC-13': 23, 'HUVEC-14': 24, 'HUVEC-15': 25, 'HUVEC-16': 26, 'HUVEC-17': 27,
            'HUVEC-18': 28, 'HUVEC-19': 29, 'HUVEC-20': 30, 'HUVEC-21': 31, 'HUVEC-22': 32, 'HUVEC-23': 33, 'HUVEC-24': 34,
            'RPE-01': 35, 'RPE-02': 36, 'RPE-03': 37, 'RPE-04': 38, 'RPE-05': 39, 'RPE-06': 40, 'RPE-07': 41, 'RPE-08': 42,
            'RPE-09': 43, 'RPE-10': 44, 'RPE-11': 45, 'U2OS-01': 46, 'U2OS-02': 47, 'U2OS-03': 48, 'U2OS-04': 49, 'U2OS-05': 50
        }
        # self.data_path = data_path
        data = pd.read_csv("/home/zhhc/mae/train.csv")
        self.data = list(data['site_id'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        exp, plate_id, wellpos, site = self.data[idx].split('_')
        for channel in range(1, 7):
            img_path = f"/media/ssd1/hc/rxrx1/images/{exp}/Plate{plate_id}/{wellpos}_s{site}_w{channel}.png"
            image = Image.open(img_path)
            image = np.array(image)
            if channel == 1:
                all_channel_img = torch.from_numpy(image).unsqueeze(0)
            else:
                all_channel_img = torch.cat((all_channel_img, torch.from_numpy(image).unsqueeze(0)), 0)

        class_id = self.class_map[exp]
        img_tensor = all_channel_img.float()
        
        # Adjust the size of the image tensor
        target_height, target_width = 224, 224  # Example target size
        img_tensor = resize_image_tensor(img_tensor, target_height, target_width)

        mean = img_tensor.mean(dim=(1, 2), keepdim=True)
        var = img_tensor.var(dim=(1, 2), keepdim=True)
        img_tensor = (img_tensor - mean) / (var + 1.e-6)**.5

        class_id_one_hot = torch.zeros(len(self.class_map), dtype=torch.float)
        class_id_one_hot[class_id] = 1.0

        return img_tensor, class_id_one_hot

def resize_image_tensor(img_tensor, target_height, target_width):
    """
    Resize a 3D image tensor (C, H, W) or 4D batch tensor (N, C, H, W) to a specified size.

    Args:
        img_tensor (torch.Tensor): Input tensor, shape should be (C, H, W) or (N, C, H, W).
        target_height (int): Target height.
        target_width (int): Target width.

    Returns:
        torch.Tensor: Resized tensor.
    """
    if img_tensor.dim() == 3:  # (C, H, W)
        img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension (1, C, H, W)
        resized = F.interpolate(img_tensor, size=(target_height, target_width), mode='bilinear', align_corners=False)
        return resized.squeeze(0)  # Remove batch dimension
    elif img_tensor.dim() == 4:  # (N, C, H, W)
        return F.interpolate(img_tensor, size=(target_height, target_width), mode='bilinear', align_corners=False)
    else:
        raise ValueError(f"Invalid tensor dimension {img_tensor.dim()}, expected 3D (C, H, W) or 4D (N, C, H, W)")

def process_resized_image(data_path, target_size=(224, 224)):
    dataset = CustomDataset(data_path=data_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for i, (images, labels) in enumerate(dataloader):
        resized_images = resize_image_tensor(images, target_size[0], target_size[1])
        print(f"Original size: {images.shape}, Resized size: {resized_images.shape}")

        assert resized_images.shape[2:] == target_size
        break  # Process only the first batch for demonstration
