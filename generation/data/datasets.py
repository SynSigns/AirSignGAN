import torch
import random
import pandas as pd
from math import floor
from torchvision import transforms
from PIL import Image

class Dataset(torch.utils.data.dataset.Dataset):
    def __init__(self, root='', batch_size=1, truncate_size=None):
        self.root = root
        self.batch_size = batch_size
        self.truncate_size = truncate_size
        
        self.data = pd.read_csv(root).values #loading csv as a numpy array
        self.data = torch.tensor(self.data, dtype=torch.float32) #converting to tensor
        if(self.truncate_size is not None):
            self.data = self.data[:self.truncate_size]

        self.data = self.data.unsqueeze(0) #adding a batch dimension
        #data is in the format [batch, timesteps, features]
        print("In the dataset class, data shape: ", self.data.shape)
        # #but we want the format to be 
        # self.data = self.data.transpose(1, 2)
        # #now the data is in the format [batch, timesteps, features]
        # print("In the dataset class, data shape after transpose: ", self.data.shape)
        # set from outside
        self.reals = None
        self.noises = None
        self.amps = None


    def __getitem__(self, index):
        amps = self.amps 
        reals = self.reals 
        noises = self.noises 

        return {'reals': reals, 'noises': noises, 'amps': amps}
       
    def __len__(self):
        return self.batch_size