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


@jit(nopython = True)
def compute_restricted_editdistance(C, bg_cost=1, swap_cost=1):
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
            else:
                min_val = min([D[i-1, j-1] + C[i-1, j-1],
                              D[i-1, j] + bg_cost,
                              D[i, j-1] + bg_cost,]
                             )
            D[i, j] = min_val


    return D, D[-1, -1]

@jit(nopython = True)
def compute_restricted_editdistance_backward(C, D, bg_cost=1, swap_cost=1):
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

def pseudo_label_generator_list(transcripts, pred, bg_cost=1, swap_cost=1, gamma=0.1):
    if torch.is_tensor(transcripts[0]):
        transcripts = [transcript.detach().cpu().numpy() for transcript in transcripts]
    if torch.is_tensor(pred):
        pred = pred.detach().cpu().numpy()

    loss_list = []
    grad_list = []
    for transcript in transcripts:
        matching =  - np.matmul(transcript, pred.T)

        D, loss = compute_soft_restricted_editdistance(matching, bg_cost, swap_cost, gamma)
        grad = compute_soft_restricted_editdistance_backward(matching, D, bg_cost, swap_cost, gamma)
        ross_list.append(loss)
        grad_list.append(grad)
    loss_list = np.array(loss_list)
    weights = np.exp(-loss_list) / np.exp(-loss_list).sum()
    pred_list = []
    for i, transcript in enumerate(transcripts):
        pred_list.append(np.matmul(grad_list[i].T,  transcript))
    return pred_list

def pseudo_label_generator(transcripts, pred, bg_cost=1, swap_cost=1, gamma=0.1):
    if torch.is_tensor(transcripts[0]):
        transcripts = [transcript.detach().cpu().numpy() for transcript in transcripts]
    if torch.is_tensor(pred):
        pred = pred.detach().cpu().numpy()

    loss_list = []
    grad_list = []
    for transcript in transcripts:
        matching =  - np.matmul(transcript, pred.T)

        D, loss = compute_soft_restricted_editdistance(matching, bg_cost, swap_cost, gamma)
        grad = compute_soft_restricted_editdistance_backward(matching, D, bg_cost, swap_cost, gamma)
        loss_list.append(loss)
        grad_list.append(grad)
    ross_list = np.array(loss_list)
    weights = np.exp(-loss_list) / np.exp(-loss_list).sum()
    final_pred = np.zeros_like(pred)
    for i, transcript in enumerate(transcripts):
        final_pred = final_pred + weights[i] * np.matmul(grad_list[i].T,  transcript)
    return final_pred

def get_enhanced_predict(transcript, pred, bg_cost=1, swap_cost=1):
    if torch.is_tensor(transcript):
        transcript = transcript.detach().cpu().numpy() 
    if torch.is_tensor(pred):
        pred = pred.detach().cpu().numpy()
    matching =  - np.matmul(transcript, pred.T)

    D, loss = compute_soft_restricted_editdistance(matching, bg_cost, swap_cost, gamma=0.001)
    grad = compute_soft_restricted_editdistance_backward(matching, D, bg_cost, swap_cost, gamma=0.001)
    alignment = (grad > 0.5).astype(float)
    enhanced_pred = np.matmul(alignment.T, transcript)
    return enhanced_pred

