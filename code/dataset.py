import os
import random
import torch
import numpy as np
import torchvision.transforms.functional as F

from torch.utils.data import Dataset
from PIL import Image
from library import load_data

class listDataset(Dataset):

    def __init__(self, root, shape = None, shuffle = True, transform = None,  train = False, seen = 0, 
                 batch_size = 1, num_workers = 4, ir_enabled = False, images_dir = None, labels_dir = None):
        
        random.shuffle(root)
        
        self.nSamples = len(root)
        self.lines = root
        self.transform = transform
        self.train = train
        self.shape = shape
        self.seen = seen
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.ir_enabled = ir_enabled
        self.images_dir = images_dir
        self.labels_dir = labels_dir
    
    # returns the length of the dataset
    def __len__(self):
        return self.nSamples

    # load and returns the next item (pair image-target) of the dataset
    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        # get next path to load
        image_path = self.lines[index]
        
        # load network input and target
        network_input_image, target = load_data(self.images_dir, self.labels_dir, image_path, self.ir_enabled)

        # apply transformation if needed (for example toTensor())
        if self.transform is not None:
            network_input_image = self.transform(network_input_image)
            
        return network_input_image, target
