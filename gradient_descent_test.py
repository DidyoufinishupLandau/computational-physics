# -*- coding: utf-8 -*-
"""

Gradient descent
@author: Pan Zhang
"""
import numpy as np
import matplotlib.pyplot as plt

x_array = np.random.randint(0,100, (100, 4)) # 100*4 matrix
parameter_simulate = np.array([[1],[2],[3],[4]]) # initialize parameter

intercept = np.random.normal(0,100,1)

y_simulate = x_array.dot(parameter_simulate) + intercept

guess_parameter = np.array([[0],[0],[0],[0]])

def cost_function(x_array, y_array, parameter):
    parameter_array = parameter

    return 1/len(x_array) * np.sum((x_array.dot(parameter_simulate) - y_array)**2)

def learning_rate_control(old_learning_rate, parameter, new_parameter):
    new_learing_rate = old_learning_rate
    if np.any(new_parameter > 10*parameter ):
        temp_learing_rate = old_learning_rate * 0.1
        new_learning_rate = temp_learing_rate
    return new_learning_rate

def gradient_descent(x_array, y_array, parameter, epoches):

    learning_rate = 0.000005
    for i in range(epoches):
        gradient = x_array.T.dot(x_array.dot(parameter) - y_array)
        parameter = parameter - 1/len(x_array)*gradient * learning_rate

    return parameter

new_parameter = gradient_descent(x_array, y_simulate, guess_parameter, 100)
cost_function(x_array, y_simulate, parameter_simulate)



