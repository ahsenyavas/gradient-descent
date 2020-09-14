# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 16:31:57 2020

@author: Ahsen Yavas
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (12.0, 9.0)


# Preprocessing Input data
data = pd.read_csv("data.csv")
X = data.iloc[:, 0]
Y = data.iloc[:, 1]

plt.scatter(X,Y)
plt.show()

#Building the model
m = 0
c = 0

L = 0.0001 # Learning rate
epoch = 1000

n = float(len(X))

# Performing Gradient Descent
for i in range(epoch):
    Y_pred = m*X + c
    d_m = (-2/n) * sum(X * (Y - Y_pred)) 
    d_c = (-2/n) * sum(Y - Y_pred)
    m = m - L * d_m
    c = c - L * d_c
   
print(m, c)

# Making predictions
Y_pred = m*X + c
plt.scatter(X, Y)
plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color='red')
plt.show()

    
    
    
    
    
