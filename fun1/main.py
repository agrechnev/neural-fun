#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 16:06:53 2018

@author: Oleksiy Grechnyev
"""
import numpy as np
import network

print("Brianna !!!")
net = network.Network([3, 4, 5, 3])
x = np.array([1., 2., 3.]).reshape(3, 1) # Column
print(x.shape)
print(net.feedforward(x))
print(x)