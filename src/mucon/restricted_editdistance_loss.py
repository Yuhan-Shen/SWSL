#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 16:30:34 2021

@author: yuhan
"""
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from numba import jit
from torch.autograd import Function
import matplotlib.pyplot as plt


@jit(nopython = True)
def compute_soft_restricted_editdistance(C, bg_cost=1, swap_cost=1, gamma=0.1):
    M, N = C.shape
    D = np.zeros((M+1, N+1))
    D[0] = np.arange(0, N+1) * bg_cost
    D[1:, 0] = np.arange(1, M+1) * bg_cost

    for i in range(1, M+1):
        for j in range(1, N+1):
            if i >= 2 and j >= 2:
                min_val = min([D[i-1, j-1] + C[i-1, j-1],
                              D[i-1, j] + bg_cost,
                              D[i, j-1] + bg_cost,
                              D[i-2, j-2] + C[i-2, j-1] + C[i-1, j-2] + bg_cost
                              ]
                             )
                sum_val = np.exp(- (D[i-1, j-1] + C[i-1, j-1] - min_val) / gamma
                           ) + np.exp(- (D[i-1, j] + bg_cost - min_val) / gamma
                           ) + np.exp(- (D[i, j-1] + bg_cost - min_val) / gamma
                           ) + np.exp(- (D[i-2, j-2] + C[i-2, j-1] + C[i-1, j-2]  + swap_cost - min_val) / gamma
                           )
            else:
                min_val = min([D[i-1, j-1] + C[i-1, j-1],
                              D[i-1, j] + bg_cost,
                              D[i, j-1] + bg_cost,]
                             )
                sum_val = np.exp(- (D[i-1, j-1] + C[i-1, j-1] - min_val) / gamma
                           ) + np.exp(- (D[i-1, j] + bg_cost - min_val) / gamma
                           ) + np.exp(- (D[i, j-1] + bg_cost - min_val) / gamma)
            soft_min_val = -gamma * np.log(sum_val) + min_val
            D[i, j] = soft_min_val


    return D, D[-1, -1]

@jit(nopython = True)
def compute_soft_restricted_editdistance_backward(C, D, bg_cost=1, swap_cost=1, gamma=0.1):
    M, N = C.shape
    grad_d = np.zeros((M+1, N+1))   ### partial L / partial d_(i, j)
    grad_d_c1 = np.zeros((M , N))   ### partial d_(i+1, j+1) / partial c_(i, j)
    grad_d_c2 = np.zeros((M , N))   ### partial d_(i+1, j+1) / partial c_(i, j-1)
                                    ### = partial d_(i+1, j+1) / partial c_(i-1, j)
    grad_d[-1, -1] = 1
    i = -1
    for j in range(N-1, -1, -1):
        b = np.exp(- (D[i, j] + bg_cost - D[i, j+1]) / gamma)
        grad_d[-1, j] = grad_d[-1, j+1] * b
    j = -1
    for i in range(M-1, -1, -1):
        x = np.exp(- (D[i, j] + bg_cost - D[i+1, j]) / gamma)
        grad_d[i, -1] = grad_d[i+1, -1] * x

    for i in range(M-1, -1, -1):
        for j in range(N-1, -1, -1):
            a = np.exp(- (D[i, j] + C[i, j] - D[i+1, j+1]) / gamma)
            b = np.exp(- (D[i, j] + bg_cost - D[i, j+1]) / gamma)
            x = np.exp(- (D[i, j] + bg_cost - D[i+1, j]) / gamma)
            if i < M-1 and j < N-1:
                y = np.exp(- (D[i, j] + C[i, j+1] + C[i+1, j]
                              + swap_cost - D[i+2, j+2]) / gamma)
                grad_d_c2[i+1, j+1] = y
                grad_d[i, j] = grad_d[i+1, j+1] * a + grad_d[i, j+1] * b + \
                    grad_d[i+1, j] * x + grad_d[i+2, j+2] * y
            else:
                grad_d[i, j] = grad_d[i+1, j+1] * a + grad_d[i, j+1] * b + grad_d[i+1, j] * x

            grad_d_c1[i, j] = a
    grad_c2 = np.zeros((M, N))
    grad_c3 = np.zeros((M, N))
    ### (partial L  * / partial d_(i+2, j+1)) * (partial d_(i+2, j+1)  * / partial d_(i, j))
    grad_c2[:-1, :]  = grad_d[2:, 1:] * grad_d_c2[1:, :]
    ### (partial L  * / partial d_(i+1, j+2)) * (partial d_(i+1, j+2)  * / partial d_(i, j))
    grad_c3[:, :-1] =  grad_d[1:, 2:] * grad_d_c2[:, 1:]

    grad_c = grad_d[1:, 1:] * grad_d_c1 + grad_c2 + grad_c3

    return grad_c

class _SoftREDLoss(Function):
    @staticmethod
    def forward(ctx, C, bg_cost, swap_cost, gamma):
        dev = C.device
        dtype = C.dtype
        bg_cost = torch.Tensor([bg_cost]).to(dev).type(dtype) # dtype fixed
        swap_cost = torch.Tensor([swap_cost]).to(dev).type(dtype) # dtype fixed
        gamma = torch.Tensor([gamma]).to(dev).type(dtype) # dtype fixed
        C_ = C.detach().cpu().numpy()
        g_ = gamma.item()
        b_ = bg_cost.item()
        s_ = swap_cost.item()
        D, softmin = compute_soft_restricted_editdistance(C_, b_, s_, g_)
        D = torch.Tensor(D).to(dev).type(dtype)
        softmin = torch.Tensor([softmin]).to(dev).type(dtype)
        ctx.save_for_backward(C, D, softmin, bg_cost, swap_cost, gamma)
        return softmin

    @staticmethod
    def backward(ctx, grad_output):
        dev = grad_output.device
        dtype = grad_output.dtype
        C, D, softmin, bg_cost, swap_cost, gamma = ctx.saved_tensors
        C_ = C.detach().cpu().numpy()
        D_ = D.detach().cpu().numpy()
        softmin_ = softmin.item()
        g_ = gamma.item()
        b_ = bg_cost.item()
        s_ = swap_cost.item()
        G = torch.Tensor(compute_soft_restricted_editdistance_backward(C_, D_, b_,s_, g_)).to(dev).type(dtype)
        return grad_output.view(-1, 1).expand_as(G) * G, None, None, None

class SoftRED_Loss(torch.nn.Module):
    def __init__(self, alpha=0.01, center_norm=False, sim='cos', threshold=2, swap_threshold=0, softmax='no', norm_by_len=True):
        super(SoftRED_Loss, self).__init__()

        self.alpha = alpha
        self.center_norm = center_norm
        self.func_apply = _SoftREDLoss.apply
        self.sim = sim
        self.threshold = threshold
        self.swap_threshold = swap_threshold if swap_threshold > 0 else threshold
        self.softmax = softmax
        self.norm_by_len = norm_by_len

    def forward(self, centers_a, centers_b):
        sorted_centers_a = centers_a
        sorted_centers_b = centers_b
        
        if self.center_norm:
            sorted_centers_a = sorted_centers_a / torch.sqrt(torch.sum(sorted_centers_a**2, axis=-1, keepdims=True) + 1e-10) 
            sorted_centers_b = sorted_centers_b / torch.sqrt(torch.sum(sorted_centers_b**2, axis=-1, keepdims=True) + 1e-10)
        
        if self.sim == 'cos':
            matching =  - torch.matmul(sorted_centers_a, sorted_centers_b.t())
        elif self.sim == 'exp':
            matching = torch.exp(-torch.matmul(sorted_centers_a, sorted_centers_b.t()))
        elif self.sim == 'euc':
            a_ = torch.unsqueeze(sorted_centers_a, 1)
            matching = torch.sum((a_ - sorted_centers_b.unsqueeze(0))**2, axis=-1)
        else:
            print('Invalid similarity metric! {}'.format(self.sim))
            raise NotImplementedError
        match_min, _ = matching.min(1)
        match_avg = match_min.mean()
        L_a, L_b = matching.shape

        if self.softmax == 'row':
            matching = F.softmax(matching, -1)
        elif self.softmax == 'col':
            matching = F.softmax(matching, 0)
        elif self.softmax == 'all':
            matching = F.softmax(matching.view(-1), 0).view(L_a, L_b)
        
        loss = self.func_apply(matching, self.threshold, self.swap_threshold, self.alpha) 
        if self.norm_by_len:
            loss = loss / L_a

        return loss 
