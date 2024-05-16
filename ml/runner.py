import optparse
import sys
import traceback
import models

import numpy as np
import matplotlib
import contextlib

from torch import nn, Tensor
import torch
import backend

def format(driver, track, sprint, year):
    vec = [0] * 56
    vec[driver] = 1
    vec[30+track] = 1
    vec[54] = sprint
    vec[55] = year
    return vec

model = models.DriverClassificationModel()
dataset = backend.DriverDataset(model)
model.train(dataset)

points = []
for i in range(30):
    points += [[0] * 24]
    


def sprint(track):
    if track == 4 or track == 5 or track == 10 or track == 18 or track == 20 or track == 22:
        return 1
    else:
        return 0
    
for driver in range(30):
    for track in range(24):
        points[driver][track] = model.run(Tensor(np.array(format(driver, track, sprint(track),1)))).detach().numpy()[0]
        
print(points)


