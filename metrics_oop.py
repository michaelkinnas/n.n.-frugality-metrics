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
    def __init__(self):
        self.records = [] # list of tuples


    def calculate(self, P_aj: float, R_aj: float, w: float) -> float:        
        return P_aj - (w / (1 + (1 / R_aj)))
    
    
    def __calculate_y_points(self, accuracy: float, resource: float) -> list[float]:
        x_points = np.linspace(0, 1, num=2)

        if accuracy < 0 or accuracy > 1:
            raise ValueError('Accuracy values must be between 0 and 1')

        if resource <= 0:
            raise ValueError('Resource values must be greater than 0')

        return [[self.calculate(accuracy, resource, x) for x in x_points], list(x_points)]
    
    
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

    
frug = Frugality()
frug.add_many([('aa', 0.50, 500), ('bb', 0.98, 600)])
frug.plot()
frug.values()
