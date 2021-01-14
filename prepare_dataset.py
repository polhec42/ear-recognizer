from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import cv2

class Rescale(object):
    """
    Rescale the image in a sample to a given size
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):

        image = sample['image']

        h,w = image.shape[:2]

        h = self.output_size[0]
        w = self.output_size[1] 

        image = cv2.resize(image, (w,h))

        return {'image': image, 'class': sample['class'], 'type': sample['type']}

class ToTensor(object):
    
    def __call__(self, sample):

        image = sample['image']

        #image = image.transpose((2,0,1)) #Uporabim če bi imel barvne slike

        image = torch.from_numpy(image)
        image = image.unsqueeze(0)
        image = image.float()

        
        return {'image': image , 'class': sample['class'], 'type': sample['type']}

class EarDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        
        self.ears_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.ears_frame)

    def __getitem__(self, index):
        """
        Function accepts index and loads image and returns it as a result. This is very memory efficient as we do not have stored all images at once in the memory.
        """

        if torch.is_tensor(index):
            index = index.tolist()
        
        image_name = os.path.join(self.root_dir, self.ears_frame.iloc[index, 1])

        # cv2 reads in BGR
        image = cv2.imread(image_name)
        b,g,r = cv2.split(image)
        image = cv2.merge([r,g,b])

        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        label = self.ears_frame.iloc[index, 2]

        data_type = str(self.ears_frame.iloc[index, 0]).split("/")[0]
        sample = {'image': image, 'class': int(label)-1, 'type': data_type}

        if self.transform:
            sample = self.transform(sample)

        return sample


def get_train_data():
    transformed_dataset = EarDataset(csv_file='data/ears/awe-train.csv', root_dir='data/ears/awe', transform=transforms.Compose([Rescale((80,40)), ToTensor()]))

    for i in range(len(transformed_dataset)):
        sample = transformed_dataset[i]

    dataloader = DataLoader(transformed_dataset, batch_size=64, shuffle=True)

    return dataloader

def get_test_data():
    transformed_dataset = EarDataset(csv_file='data/ears/awe-test.csv', root_dir='data/ears/awe', transform=transforms.Compose([Rescale((80,40)), ToTensor()]))

    for i in range(len(transformed_dataset)):
        sample = transformed_dataset[i]

    dataloader = DataLoader(transformed_dataset, batch_size=64, shuffle=True)

    return dataloader

if __name__ == "__main__":
    
    """
    ear_dataset = EarDataset(csv_file='data/ears/awe-translation.csv', root_dir='data/ears/awe')

    fig = plt.figure()

    for i in range(len(ear_dataset)):
        sample = ear_dataset[i]

        scale = Rescale((36,24))
        transformed_sample = scale(sample)

        print(i, sample['image'].shape, sample['class'], sample['type'])

        ax = plt.subplot(1, 4, i + 1)
        plt.tight_layout()
        ax.set_title('Sample #{}'.format(i))
        ax.axis('off')
        plt.imshow(transformed_sample['image'])

        if i == 3:
            plt.show()
    """

    
"""
https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
https://towardsdatascience.com/handwritten-digit-mnist-pytorch-977b5338e627
"""