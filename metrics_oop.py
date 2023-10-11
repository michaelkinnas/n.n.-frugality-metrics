"""
A module that contains tools for calculating frugality metrics, intended for neural network algorithms.
"""
import matplotlib.pyplot as plt
import numpy as np
from typing import Iterable
from torch.nn import Module
from torch.optim import Optimizer
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader
import torch

class Frugality():
    """ 
    Calculates frugality scores of neural network algorithms given prediction accuracy and resource consumption.
    """
    def __init__(self):
        self.records = [] # list of tuples


    def calculate(self, P_aj: float, R_aj: float, w: float) -> float:        
        return P_aj - (w / (1 + (1 / R_aj)))
        # return P_aj - (1 / (1 + (np.exp(-np.log(R_aj)))))
    
    
    def __calculate_y_points(self, accuracy: float, resource: float) -> list[float]:
        x_points = np.linspace(0, 1, num=2)

        if accuracy < 0 or accuracy > 1:
            raise ValueError('Accuracy values must be between 0 and 1')

        if resource <= 0:
            raise ValueError('Resource values must be greater than 0')

        return [list(x_points), [self.calculate(accuracy, resource, x) for x in x_points]]
    
    
    def add(self, label: str, accuracy: float, resources: float):
        x, y = self.__calculate_y_points(accuracy, resources)      
        self.records.append((label, accuracy, resources, [x, y]))
    
    
    def add_many(self, values: Iterable[tuple[str, float, float]]):
        for item in values:
            label, accuracy, resource = item
            self.add(label, accuracy, resource)

    
    def plot(self):
        plt.figure(figsize=(12,8))
        plt.title('Frugality Lines')
        plt.xlabel('Index w')
        plt.ylabel('Frugality score')
        plt.xlim([0, 1])
        plt.ylim([0, 1])

        for label, (x, y) in [a[0:4:3] for a in self.records]:
            plt.plot(x, y, label=label)

        plt.legend()
        plt.show()


    def values(self):
        for item in self.records:
            label, acc, res = item[:3]
            print(f'label: {label}, accuracy: {acc}, resources: {res}')

# For a given model keep track:

# Time taken training
# Accuracy for every epoch
# Loss function for every epoch
# 

"""
aaa = TrainingMetric([(model, optimizer, loss_function, dataloader, epochs, seed, label)])
"""

class TrainingMetrics:
    def __init__(self, models: Iterable[tuple[Module, _Loss, Optimizer, DataLoader, int, int, str]]):
        self.models = models # array of tuples

    def train(self, models: Iterable[str] | None = None): # Train models acording passed labels or tain all if None was passed
        pass
    
    #TEST HITS
    def train(self, index: int | None = None):
        pass

    def metrics(self, models: Iterable[str] | None = None):
        pass

    def plot(self, models: Iterable[str] | None = None):
        pass

    def models(self, models: Iterable[str] | None = None):
        pass



# loss_fn = torch.nn.BCEWithLogitsLoss()
# print(loss_fn)
    

