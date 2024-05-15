import optparse
import sys
import traceback
import models

def main():
    model = models.DriverClassificationModel()
    dataset = backend.DriverDataset(model)
    model.train(dataset)