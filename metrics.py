"""
A module that contains functions for calculating frugality metrics, intended for neural network algorithms.
"""

__author__ = 'Michael Kinnas'

import matplotlib.pyplot as plt
import numpy as np
from typing import Iterable

def A3R(S_aref: float, S_aj: float, T_aref: int, T_aj: int, N: int) -> float:
    """
    A3R multi objective measure function.

    Parameters
    ----------           
    S_aref : Accuracy of reference algorithm aref
    S_aj :   Accuracy of algorithm aj
    T_aref : Run time of reference algorithm aref in seconds
    T_aj :   Run time of algorithm aj in seconds
    N :      Influence of time parameter

    """
    if type(N) is not int:
        raise TypeError(f'N parameter must be of type int. Got {type(N)} instead')
    
    return (S_aj / S_aref) / (T_aj / T_aref) ** (1 / N) # same as Nth root of (T_aj / T_aref)


def A3Rs(S_aj: float, T_aj: int, N: int) -> float:
    """
    Simplified version of A3R function. 
    S_aref and T_aref variables are assumed of having the value of 1.
    
    Inputs:           
        S_aj:   Accuracy of algorithm aj            
        T_aj:   Run time of algorithm aj in seconds
        N:      Influence of time parameter

    """
    if type(N) is not int:
        raise TypeError(f'N parameter must be of type int. Got {type(N)} instead')
    
    return S_aj / T_aj ** (1 / N) # same as Nth root of T_aj


def frug(P_aj: float, R_aj: float, w: float) -> float:
    """       
    Calculates a frugality score of an algorithm with respect to resource consumption and predictive performance.

    Parameters
    ----------
        P_aj: Predictive performance of algorithm aj. (* from paper: ROC or AUC (for binary class problems?) | AUC for multiclass problems)
        R_aj: Resource consumption of algorithm aj. (*from paper: CPU time in milliseconds and non zero)
        w:    Weight coefficient of frugality. (*from paper: 0 and 1 but can be higher)
    
    Extra notes from paper:
    ----------
        P_aj: AUC (0 to 1)
        R_aj: T_train + T_test,
        w: Values higher than 1 indicate predictive performance is less important

    Question:
    ----------
        For my thesis can `R_aj` be power consumption?
    """
    if P_aj < 0 or P_aj > 1:
        raise ValueError('P_aj value must be between 0 and 1')  
    
    if R_aj <= 0:
        raise ValueError('R_aj value must be a positive number and non 0')
    
    if w < 0:
        raise ValueError('w value must be 0 or greater')

    return P_aj - (w / (1 + (1 / R_aj)))


def plot_frugality_lines(names: Iterable[str], accuracy: Iterable[float], resource: Iterable[float], score_function=frug):
    """
    Plot frugality lines for comparisson.

    Parameters
    ----------
    names: An iterable of each function name, for the plot legend.
    accuracy: An iterable of accuracy values. Must be between 0 and 1.
    resource: An iterable of resource cost values. Must be greater than 0.
    score_fn: The score calculation function, `frug` by default.
    """

    # Exception handling
    if len(accuracy) != len(resource) or len(accuracy) != len(names):
        raise ValueError('names, accuracy and resource must be the same size')    
    
    for num in accuracy:
        if num < 0 or num > 1:
            raise ValueError('Accuracy values must be between 0 and 1')
    
    for num in resource:
        if num <= 0:
            raise ValueError('Resource values must be greater than 0')        

    # Add names to empty strings
    for idx, name in enumerate(names):
        if name == '':
            names[idx] = '(' + str(idx) + ')'

    n_points = 2
    x_points = np.linspace(0, 1, n_points)
    frugality_scores = []
    line_labels = []

    for acc, res in zip(accuracy, resource):
        y_points = []
        line_labels.append(acc)
        for x in x_points:
            y_points.append(score_function(acc, res, x))
        frugality_scores.append(y_points)

    plt.figure(figsize=(12,8))
    plt.title('Frugality Lines')
    plt.xlabel('Index w')
    plt.ylabel('Frugality score')    
    plt.xlim([0, 1])
    plt.ylim([0, 1])

    for score, name in zip(frugality_scores, names):      
        plt.plot(x_points, score, label=name)

    plt.legend()
    plt.show()