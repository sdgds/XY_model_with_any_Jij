#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
sys.path.append('/Users/mac/Desktop/XY_model_and_Neural_maps/XY-MODEL')
from XY_model import XYSystem
import BrainSOM

# 示例
L,N = 10,100
nbr = {i : ((i // L) * L + (i + 1) % L, (i + L) % N,
        (i // L) * L + (i - 1) % L, (i - L) % N) \
                            for i in list(range(N))}
J = np.zeros((100, 100))
for i in range(100):
    J[i][list(nbr[i])] = 1
xy_system_1 = XYSystem(J, temperature = 1.5, width = 10)
xy_system_1.show()
print('Energy per spin:%.3f'%xy_system_1.energy)

xy_system_1.equilibrate(show=True)
xy_system_1.show()

config_matrix = xy_system_1.list2matrix(xy_system_1.spin_config)
U = np.cos(config_matrix)
V = np.sin(config_matrix)
theta_range = np.linspace(0,2*np.pi,256)
colorbar = mpl.cm.nipy_spectral(np.arange(256))
color_map = mpl.colors.ListedColormap(colorbar, name='myColorMap', N=colorbar.shape[0])
plt.figure(figsize=(8,8))
for i in tqdm(range(10)):
    for j in range(10):  
        a,b = U[i,j],V[i,j]
        if b>0 and a>0:
            theta = np.arctan(b/a)
        if b>0 and a<0:
            theta = np.pi + np.arctan(b/a)
        if b<0 and a<0:
            theta = np.pi + np.arctan(b/a)
        if b<0 and a>0:
            theta = 2*np.pi + np.arctan(b/a) 
        d = np.abs(theta_range-theta)
        fig = plt.scatter(i, j, marker=',',
                          color=color_map(np.where(d==d.min())[0]), 
                          alpha=1)
        plt.xlim([0,10])
        plt.ylim([10,0])        




# 实例(SOM)
som = BrainSOM.VTCSOM(64, 64, 1000, sigma=4, learning_rate=0.5,
                      sigma_decay_speed=2, lr_decay_speed=2,
                      neighborhood_function='gaussian', random_seed=None)
som._weights = np.load('/Users/mac/Desktop/TDCNN/Results/Alexnet_fc8_SOM/pca_init.npy')

J = np.corrcoef(som._weights.reshape(-1,1000))
xy_system_1 = XYSystem(J, temperature = 5, width = 64)
xy_system_1.show()
print('Energy per spin:%.3f'%xy_system_1.energy)

xy_system_1.equilibrate(show=True)

colorbar = mpl.cm.nipy_spectral(np.arange(0,256))
color_map = mpl.colors.ListedColormap(colorbar, name='myColorMap', N=colorbar.shape[0])
xy_system_1.show()

config_matrix = xy_system_1.list2matrix(xy_system_1.spin_config)
U = np.cos(config_matrix)
V = np.sin(config_matrix)
theta_range = np.linspace(0,2*np.pi,256)
colorbar = mpl.cm.nipy_spectral(np.arange(256))
color_map = mpl.colors.ListedColormap(colorbar, name='myColorMap', N=colorbar.shape[0])
plt.figure(figsize=(8,8))
for i in tqdm(range(64)):
    for j in range(64):  
        a,b = U[i,j],V[i,j]
        if b>0 and a>0:
            theta = np.arctan(b/a)
        if b>0 and a<0:
            theta = np.pi + np.arctan(b/a)
        if b<0 and a<0:
            theta = np.pi + np.arctan(b/a)
        if b<0 and a>0:
            theta = 2*np.pi + np.arctan(b/a) 
        d = np.abs(theta_range-theta)
        fig = plt.scatter(i, j, marker=',',
                          color=color_map(np.where(d==d.min())[0]), 
                          alpha=1)
        plt.xlim([0,64])
        plt.ylim([64,0])
        
        
        
        
# 实例(Hebb-SOM)
som = BrainSOM.VTCSOM(64, 64, 4096, sigma=3, learning_rate=0.5, 
                      sigma_decay_speed=2, lr_decay_speed=2,
                      neighborhood_function='gaussian')
som._weights = np.load('/Users/mac/Desktop/TDCNN/Results/Alexnet_fc6_SOM/Alexnet_fc6_pca_init_longinh_shortexc.npy')
som.excitory_value = np.load('/Users/mac/Desktop/TDCNN/Results/Alexnet_fc6_SOM/Alexnet_fc6_pca_init_excitory_value.npy')
som.inhibition_value = np.load('/Users/mac/Desktop/TDCNN/Results/Alexnet_fc6_SOM/Alexnet_fc6_pca_init_inhibition_value.npy')

J = som.excitory_value - som.inhibition_value
J = J.reshape(4096,4096)
xy_system_1 = XYSystem(J, temperature = 1.5, width = 64)
xy_system_1.show()
print('Energy per spin:%.3f'%xy_system_1.energy)

xy_system_1.equilibrate(show=True)

colorbar = mpl.cm.nipy_spectral(np.arange(0,256))
color_map = mpl.colors.ListedColormap(colorbar, name='myColorMap', N=colorbar.shape[0])
xy_system_1.show()

config_matrix = xy_system_1.list2matrix(xy_system_1.spin_config)
U = np.cos(config_matrix)
V = np.sin(config_matrix)
theta_range = np.linspace(0,2*np.pi,256)
colorbar = mpl.cm.nipy_spectral(np.arange(256))
color_map = mpl.colors.ListedColormap(colorbar, name='myColorMap', N=colorbar.shape[0])
plt.figure(figsize=(8,8))
for i in tqdm(range(64)):
    for j in range(64):  
        a,b = U[i,j],V[i,j]
        if b>0 and a>0:
            theta = np.arctan(b/a)
        if b>0 and a<0:
            theta = np.pi + np.arctan(b/a)
        if b<0 and a<0:
            theta = np.pi + np.arctan(b/a)
        if b<0 and a>0:
            theta = 2*np.pi + np.arctan(b/a) 
        d = np.abs(theta_range-theta)
        fig = plt.scatter(i, j, marker=',',
                          color=color_map(np.where(d==d.min())[0]), 
                          alpha=1)
        plt.xlim([0,64])
        plt.ylim([64,0])
        
        