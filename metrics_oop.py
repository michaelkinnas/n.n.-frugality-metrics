"""
A module that contains functions for calculating frugality metrics, intended for neural network algorithms.
"""

__author__ = 'Michael Kinnas'

import matplotlib.pyplot as plt
import numpy as np
from typing import Iterable

class Frugality():
    """ 
    Calculates frugality scores of neural network algorithms given prediction accuracy and resource consumption.
    """
    def __init__(self, values: Iterable[tuple[str, float, float]] = None):
        self.x_points = np.linspace(0, 1, num=2)

        self.values = [values] # list of tuples
        self.ys = [] #list of lists of calculated y points for each tuple
        self.labels = []

        if values != None:
            self.add_many(values)
       
        
    def calculate(self, P_aj: float, R_aj: float, w: float) -> float:        
        return P_aj - (w / (1 + (1 / R_aj)))
    
    
    def calculate_y_points(self, accuracy: float, resource: float) -> list[float]:
        if accuracy < 0 or accuracy > 1:
            raise ValueError('Accuracy values must be between 0 and 1')

        if resource <= 0:
            raise ValueError('Resource values must be greater than 0')

        return [self.calculate(accuracy, resource, x) for x in self.x_points]
    
    
    def add(self, name: str, accuracy: float, resources: float):
        self.ys.append(self.calculate_y_points(accuracy, resources))
        self.labels.append(name)
    
    
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

        for score, name in zip(self.ys, self.labels):
            plt.plot(self.x_points, score, label=name)

        plt.legend()
        plt.show()