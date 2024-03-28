import os
import random
import torch

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ContourDiffDataset(Dataset):
    def __init__(self, df_meta, image_directory, contour_directory, transform_img=None, transform_contour=None, generator_seed=None, config=None):
        self.df_meta = df_meta
        self.image_directory = image_directory
        self.contour_directory = contour_directory
        
        self.transform_img = transform_img
        self.transform_contour = transform_contour
        self.generator_seed = generator_seed
        if self.generator_seed is not None:
            self.seed_generator = torch.Generator().manual_seed(self.generator_seed)
        self.length = self.df_meta.shape[0]
        self.config = config

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        img_name = self.df_meta.iloc[index, :]["image_name"]
        if self.config is None or self.config.in_channels == 1:
            img = Image.open(os.path.join(self.image_directory, img_name)).convert("L")
        elif self.config.in_channels == 3:
            img = Image.open(os.path.join(self.image_directory, img_name)).convert("RGB")

        contour_name = self.df_meta.iloc[index, :]["contour_name"]
        contour = Image.open(os.path.join(self.contour_directory, contour_name))
        
        if self.generator_seed is not None:
            seed = self.seed_generator.seed()
    
        if self.transform_img is not None:
            if self.generator_seed is not None:
                torch.manual_seed(seed)
            img = self.transform_img(img)
            
        if self.transform_contour is not None:
            if self.generator_seed is not None:
                torch.manual_seed(seed)
            contour = self.transform_contour(contour)      

        return {
            "images": img,
            "contours": contour,
            "image_name": img_name,
            "contour_name": contour_name
        }