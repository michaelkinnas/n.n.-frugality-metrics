"""
A module that contains functions for calculating frugality metrics, intended for neural network algorithms.
"""

__author__ = 'Michael Kinnas'

import matplotlib.pyplot as plt
import numpy as np
from typing import Iterable
from torch.nn import Module
from torch.optim import Optimizer
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader
from timeit import default_timer as timer
from tqdm.notebook import tqdm
import torch


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



def plot_frugality_lines(values: Iterable[tuple[str, float, float]], score_function=frug):
    """
    Plot frugality lines for comparisson.

    Parameters
    ----------
    names: An iterable of each function name, for the plot legend.
    accuracy: An iterable of accuracy values. Must be between 0 and 1.
    resource: An iterable of resource cost values. Must be greater than 0.
    score_fn: The score calculation function, `frug` by default.
    """
    n_points = 2
    x_points = np.linspace(0, 1, n_points)   

    plt.figure(figsize=(12,8))
    plt.title('Frugality Lines')
    plt.xlabel('Index w')
    plt.ylabel('Frugality score')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    
    for idx, item in enumerate(values):
        accuracy, resources, label = item        
        plt.plot(x_points, [score_function(accuracy, resources, x) for x in x_points], label=label)       

    plt.legend()
    plt.show()

def accuracy():
    pass


def train_metrics(model: Module,                  
                  loss_fn: _Loss, 
                  optimizer: Optimizer, 
                  dataloader: DataLoader, 
                  epochs: int, 
                  seed: int, 
                  device='cpu', 
                  accuracy_fn=accuracy):
    
    accuracy = []
    losses = []

    
    X_train, y_train, X_test, y_test = X_train.to(device), y_train.to(device), X_test.to(device), y_test.to(device)

    start_time = timer()
    for epoch in tqdm(range(epochs)):
        model.train()

        # 1. Forward passs
        y_train_logits = model(X_train)
        y_train_probs = torch.sigmoid(y_train_logits)
        y_train_pred = torch.round(y_train_probs) # logits -> prediction probabilities -> prediction labels


        # 2. Calculate the loss
        loss = loss_fn(y_train_logits, y_train) # BCEWithLogitsLoss (takes in logits as first input)
        acc = accuracy_fn(y_true=y_train, y_pred=y_train_pred)

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Step the optimizer
        optimizer.step()

        
        
        if epoch % 50 == 0:
            print(f'Epoch: {epoch:04d} | Loss: {loss:.4f}, Acc: {acc:.2f}%')


    end_time = timer()
    time = start_time - end_time
    # print_train_time(start=start_time, end=end_time, device=device)


    return accuracy, losses, time
    return {
        'accuracy': [],
        'loss': [],
        'time': time
    }