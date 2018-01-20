# encoding: utf-8

"""
Read images and corresponding labels.
"""

import os
import torch
import numpy as np

from PIL import Image


class ChestXrayDataSet(torch.utils.data.Dataset):
    def __init__(self, data_dir, image_list, transform=None):
        """
        Args:
            data_dir: path to image directory.
            image_list: path to the file containing images
                with corresponding labels.
            transform: optional transform to be applied on a sample.
        """

        image_names = []
        labels = []

        if isinstance(image_list, (list, tuple)):
            image_names = image_list
            labels = np.zeros([1, 14])
        else:
            with open(image_list, "r") as f:
                for line in f:
                    items = line.split()
                    image_name = items[0]
                    label = items[1:]
                    label = [int(i) for i in label]
                    image_name = os.path.join(data_dir, image_name)
                    image_names.append(image_name)
                    labels.append(label)

        self.image_names = image_names
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index: the index of item

        Returns:
            image and its labels
        """
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.FloatTensor(label).cpu()

    def __len__(self):
        return len(self.image_names)

