# data_util.py
import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
# PIL Image import might not be used if cv2 is primary, but keep if other parts use it
# import io
# from PIL import Image


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, mask):
        image = (image - self.mean) / self.std
        if mask.max() > 1.0: # Normalize mask if it's not already 0-1
            mask /= 255.0
        return image, mask

class RandomCrop(object):
    def __call__(self, image, mask):
        H, W, _ = image.shape
        # Ensure crop dimensions are valid
        randw_max = W // 8
        randh_max = H // 8
        if randw_max == 0 or randh_max == 0: # Cannot crop if image is too small
             return image, mask

        randw = np.random.randint(randw_max) if randw_max > 0 else 0
        randh = np.random.randint(randh_max) if randh_max > 0 else 0
        
        offseth = 0 if randh == 0 else np.random.randint(randh)
        offsetw = 0 if randw == 0 else np.random.randint(randw)
        
        p0, p1, p2, p3 = offseth, H + offseth - randh, offsetw, W + offsetw - randw
        return image[p0:p1, p2:p3, :], mask[p0:p1, p2:p3]

class RandomFlip(object):
    def __call__(self, image, mask):
        if np.random.randint(2) == 0:
            return image[:, ::-1, :], mask[:, ::-1]
        else:
            return image, mask

class Resize(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, image, mask):
        image = cv2.resize(image, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR) # Changed to INTER_LINEAR for images
        mask = cv2.resize(mask, dsize=(self.W, self.H), interpolation=cv2.INTER_NEAREST) # NEAREST for masks
        return image, mask

class ToTensor(object): # For test/val
    def __call__(self, image, mask):
        image_tensor = torch.from_numpy(image.copy()).float() # Add .copy() for safety
        image_tensor = image_tensor.permute(2, 0, 1)
        mask_tensor = torch.from_numpy(mask.copy()).float() # Add .copy() for safety
        return image_tensor, mask_tensor

class ToTensor1(object): # For train, mask has extra dim
    def __call__(self, image, mask):
        image_tensor = torch.from_numpy(image.copy()).float()
        image_tensor = image_tensor.permute(2, 0, 1)
        mask_tensor = torch.from_numpy(mask.copy()).float().unsqueeze(0)
        return image_tensor, mask_tensor

class Config(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        # These mean/std values should be appropriate for your dataset
        self.mean   = np.array([[[124.55, 118.90, 102.94]]]) # Example values
        self.std    = np.array([[[ 56.77,  55.97,  57.50]]])  # Example values
        
        # 定义文件名前缀和扩展名
        self.source_image_prefix = "source_image_"
        self.mask_image_prefix = "mask_image_"  # <--- 修改此处
        self.file_extension = ".png"

    def __getattr__(self, name):
        if name in self.kwargs:
            return self.kwargs[name]
        elif hasattr(self, name): # 确保可以访问到上面定义的属性
            return getattr(self, name)
        else:
            # 为了避免在Config未定义某些属性时出现AttributeError，可以返回None或抛出更明确的错误
            # print(f"Warning: Attribute '{name}' not found in Config.kwargs or as a direct attribute.")
            return None



class Data(Dataset):
    def __init__(self, cfg):
        self.cfg        = cfg
        self.normalize  = Normalize(mean=cfg.mean, std=cfg.std)
        self.randomcrop = RandomCrop()
        self.randomflip = RandomFlip()
        self.resize     = Resize(384, 384) # Or other desired size
        
        # Choose ToTensor variant based on mode or a config flag if necessary
        # self.totensor   = ToTensor() # Typically for test/val
        # self.totensor1  = ToTensor1() # Typically for train

        txt_file_path = os.path.join(cfg.datapath, cfg.mode + '.txt')
        with open(txt_file_path, 'r') as lines:
            self.samples = [line.strip() for line in lines if line.strip()]
        if not self.samples:
            print(f"Warning: No samples loaded from {txt_file_path}. Check the file and its content.")


    def __getitem__(self, idx):
        common_identifier = self.samples[idx] # e.g., "00000_414"

        source_filename = self.cfg.source_image_prefix + common_identifier + self.cfg.file_extension
        mask_filename = self.cfg.mask_image_prefix + common_identifier + self.cfg.file_extension

        image_path = os.path.join(self.cfg.datapath, self.cfg.mode, 'source_image', source_filename)
        mask_path  = os.path.join(self.cfg.datapath, self.cfg.mode, 'mask_image', mask_filename)

        try:
            image = cv2.imread(image_path)
            if image is None:
                raise FileNotFoundError(f"Image not found or cv2.imread failed: {image_path}")
            image = image[:,:,::-1].astype(np.float32) # BGR to RGB

            mask = cv2.imread(mask_path, 0) # Read mask as grayscale
            if mask is None:
                raise FileNotFoundError(f"Mask not found or cv2.imread failed: {mask_path}")
            mask = mask.astype(np.float32)
        except Exception as e:
            print(f"Error loading data for identifier: {common_identifier}")
            print(f"Attempted image path: {image_path}")
            print(f"Attempted mask path: {mask_path}")
            raise e

        original_shape = mask.shape # H, W

        if self.cfg.mode == 'train':
            image, mask = self.normalize(image, mask)
            image, mask = self.randomcrop(image, mask)
            image, mask = self.randomflip(image, mask)
            image, mask = self.resize(image, mask) # Resize after augmentations
            totensor_transform = ToTensor1() # for train
            image, mask = totensor_transform(image, mask)
            return image, mask
        else: # 'test' or 'val'
            image, mask = self.normalize(image, mask)
            image, mask = self.resize(image, mask) # Resize before to tensor
            totensor_transform = ToTensor() # for test/val
            image, mask = totensor_transform(image, mask)
            return image, mask, original_shape, common_identifier # common_identifier is more useful than full name here

    def __len__(self):
        return len(self.samples)

    # The collate function from your original code had issues.
    # If resizing is consistently done in __getitem__, DataLoader's default collate_fn
    # should work for batching tensors, unless you have specific padding needs for variable sizes.
    # If you need a custom collate_fn, it should be adapted to handle the output of __getitem__.
    # For simplicity, I'm removing the old collate and assuming default_collate or a new one will be used.
    # def collate(self, batch):
    #     ... (this needs careful reimplementation if custom behavior is needed)