import collections
import os
import time
import os
import csv

import matplotlib.pyplot as plt
import numpy as np

from torch import nn
import torch
from torch.utils.data import Dataset, DataLoader

def format(driver, track, sprint, year):
    vec = [0] * 56
    vec[driver] = 1
    vec[30+track] = 1
    vec[54] = sprint
    vec[55] = year
    return vec

class Custom_Dataset(Dataset):
    def __init__(self, x, y, transform=None):
        # assert isinstance(x, np.ndarray)
        # assert isinstance(y, np.ndarray)
        # assert np.issubdtype(x.dtype, np.floating)
        # assert np.issubdtype(y.dtype, np.floating)
        # assert x.ndim == 2
        # assert y.ndim == 2
        # assert x.shape[0] == y.shape[0]
        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        label = self.y[idx]
        x = self.x[idx]
         
        sample = {'x': torch.Tensor(x), 'label': torch.Tensor(label)}

        if self.transform:
            sample = self.transform(sample)
        
        return sample
    


    def get_validation_accuracy(self):
        raise NotImplementedError(
            "No validation data is available for this dataset. "
            "In this assignment, only the Digit Classification and Language "
            "Identification datasets have validation data.")

class DriverDataset(Custom_Dataset):
    def __init__(self, model):
        points = 500
        x = []
        y = []
        
        drivers = {
            "ver": 0, 
            "per": 1, 
            "lec": 2, 
            "sai": 3, 
            "nor": 4, 
            "pia": 5, 
            "rus": 6, 
            "ham": 7, 
            "alo": 8, 
            "str": 9, 
            "tsu": 10, 
            "ric": 11, 
            "oco": 12, 
            "gas": 13, 
            "hul": 14, 
            "mag": 15, 
            "alb": 16, 
            "sar": 17, 
            "bot": 18, 
            "zho": 19, 
            "red": 20, 
            "fer": 21, 
            "mcl": 22, 
            "mer": 23,
            "ast": 24, 
            "rac": 25, 
            "haa": 26, 
            "alp": 27, 
            "wil": 28, 
            "kic": 29
        }
        
        tracks = {
            "bah": 0,
            "jed": 1, 
            "alb": 2, 
            "suz": 3, 
            "sha": 4, 
            "mia": 5, 
            "imo": 6, 
            "mon": 7, 
            "can": 8, 
            "bar": 9, 
            "red": 10, 
            "sil": 11, 
            "hun": 12, 
            "spa": 13, 
            "zan": 14, 
            "mza": 15, 
            "bak": 16, 
            "sin": 17, 
            "cot": 18, 
            "mex": 19, 
            "sao": 20, 
            "vgs": 21, 
            "qat": 22, 
            "abu": 23
        }
        
        with open("ml/data/f1 scores - Sheet1.csv", 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            fields = next(csvreader)
            for row in csvreader:
                x.append(format(drivers[row[1]], tracks[row[2]], int(row[3]), int(row[4])))
                y.append(float(row[5]))
            
        super().__init__(np.array(x), np.expand_dims(y, axis=1))

        self.model = model


        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = self.x[idx]
        y = self.y[idx]

        return {'x': torch.tensor(x, dtype=torch.float32), 'label': torch.tensor(y, dtype=torch.float32)}
    
