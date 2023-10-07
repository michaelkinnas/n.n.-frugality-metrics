"""
A module that contains functions for calculating frugality metrics, intended for neural network algorithms.
"""

__author__ = 'Michael Kinnas'

import numpy as np

class FrugalityMetrics():

    def __init__(self):
        return
    
    def A3R(self, S_aref: float, S_aj: float, T_aref: int, T_aj: int, N: int) -> float:
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
       
        if type(T_aref) != 'float':
            raise TypeError(f'Expected int. Got {type(T_aref)} insted.')        

        return (S_aj / S_aref) / np.power((T_aj / T_aref), 1 / N)
    
    
    def A3Rs(self, S_aj: float, T_aj: int, N: int) -> float:
        """
        Simplified version of A3R function. 
        S_aref and T_aref variables are assumed of having the value of 1.
        
        Inputs:           
            S_aj:   Accuracy of algorithm aj            
            T_aj:   Run time of algorithm aj in seconds
            N:      Influence of time parameter

        """
  
        # if not isinstance(N, int):
        if type(N) is not int:
            raise TypeError(f'N parameter must be of type int. Got {type(N)} instead')
        
        return S_aj / np.power(T_aj, 1 / N)
    
    
    def frug(self, P_aj: float, R_aj: float, w: float) -> float:
        """       
        Calculates a frugality score of an algorithm with respect to resource consumption and predictive performance.

        Parameters
        ----------
            P_aj: Predictive performance of algorithm aj. (* from paper: ROC or AUC (for binary class problems?) | AUC for multiclass problems)
            R_aj: Resource consumption of algorithm aj. (*from paper: CPU time in milliseconds and non zero)
            w:    Weight coefficient of frugality. (*from paper: 0 and 1 but can be higher)
        
        Extra notes from paper:
        ----------
            P_aj: AUC
            R_aj: T_train + T_test,
            w: Values higher than 1 indicate predictive performance is less important

        Question:
        ----------
            For my thesis can `R_aj` be power consumption?
        """
        
        if w < 0:
            raise ValueError('w value must be 0 or greater')
        
        if R_aj <= 0:
            raise ValueError('R_aj value must be a positive number and non 0')

        return P_aj - (w / (1 + (1 / R_aj)))
    
 
metrics = FrugalityMetrics()
metrics.A3Rs(0.97, 26, 1)

