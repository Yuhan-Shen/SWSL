import numpy as np
import torch

import numpy as np
import scipy.optimize
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
from numba import jit

@jit(nopython = True)
def compute_ftp_cost_matrix(max_n_seg, T, V):
    L = np.zeros((max_n_seg, T))
    L[0] = V[0]
    for i in range(1, max_n_seg):
        for j in range(i+1, T):
            L[i, j] = (L[i-1, i:j] + V[i+1:j+1, j]).min()
    return L

class FTP_Layer(nn.Module):
    ###  Flexible Transcript Prediction Layer
    def __init__(self, min_n_seg, max_n_seg, penalty_weight=1, dtype=torch.float, device='cuda', use_avg_prob=False):
        super(FTP_Layer, self).__init__()
        self.min_n_seg = min_n_seg
        self.max_n_seg = max_n_seg
        self.penalty_weight = penalty_weight
        self.dtype = dtype
        self.device = device
        self.use_avg_prob = use_avg_prob

    def forward(self, X, given_actions=None):
        """
        X: 2D-tensor, T x K, log likelihood
        """
        T, K = X.shape
        if given_actions is not None:
            # if given_actions, mask the other actions
            mask = torch.ones_like(X) * 10000
            mask[:, given_actions] = 1
            masked_X = mask * X
            # compute the cumulative sum of negative log-likelihood
            X_cum = torch.cumsum(-masked_X.detach(), axis=0)
        else:
            # compute the cumulative sum of negative log-likelihood
            X_cum = torch.cumsum(-X.detach(), axis=0)
        X_cum_shift = torch.zeros_like(X_cum)
        X_cum_shift[1:] = X_cum[:-1]
        U, _ = (X_cum.unsqueeze(0) - X_cum_shift.unsqueeze(1)).min(-1)
        V = torch.triu(U, 1).detach().cpu().numpy()
        if self.use_avg_prob:
            dur_mask = np.arange(T).reshape([1, -1]) - np.arange(-1, T-1).reshape([-1, 1])
            V = V / np.maximum(dur_mask, 1)
        del U
    
        max_n_seg = min(self.max_n_seg, T)
        min_n_seg = min(self.min_n_seg, T)
        L = compute_ftp_cost_matrix(max_n_seg, T, V)
        
        Loss = L[min_n_seg-1:, -1] 
        Loss = Loss + np.arange(min_n_seg, max_n_seg+1) * self.penalty_weight * T
        opt_n_seg = Loss.argmin() + min_n_seg - 1
                  
        ### back-tracking
        change_points = [T-1]
        opt_t = T-1
        for i in range(opt_n_seg, 0, -1):
            opt_t = (L[i-1, :opt_t] + V[1:opt_t+1, opt_t]).argmin()
            change_points.append(opt_t)
            if opt_t == 0:
                break
               
        change_points.reverse()
        prototypes = self.generate_prototypes(X, change_points)
        if given_actions is not None:
            mask = torch.ones_like(prototypes) * 10000
            mask[:, given_actions] = 1
            masked_prototypes = mask * prototypes
            return masked_prototypes
        return prototypes
    
    
    def generate_prototypes(self, X, change_points):
        # compute the average class-wise probability of each segment based on the change points from FTP
        change_points.insert(0, -1)
        segments = [(change_points[i]+1, change_points[i+1]+1) for i in range(len(change_points)-1)]
        centers = []
        for segment in segments:
            center = X[segment[0]: segment[1]].mean(0)
            centers.append(center)
        return torch.stack(centers)
