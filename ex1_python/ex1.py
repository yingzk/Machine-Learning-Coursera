#!usr/bin/env python  
# -*- coding:utf-8 -*-

""" 
@author:yzk13 
@time: 2018/05/02 
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def warmUpExercise():
    A = np.eye(5)
    return A


def computeCost(X, y, theta):
    m = len(y)
    squaredError = (X.dot(theta).reshape(m, ) - y) ** 2
    J = sum(squaredError) / (2 * m)
    return J


def gradientDescent(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = np.zeros((num_iters, 1))

    for iter in range(num_iters):
        theta_0 = theta[0] - alpha / m * sum((X.dot(theta).reshape(m, ) - y) * X[:, 0])
        theta_1 = theta[1] - alpha / m * sum((X.dot(theta).reshape(m, ) - y) * X[:, 1])
        theta = np.array([theta_0, theta_1])
        J_history[iter] = computeCost(X, y, theta)
    return theta, J_history


if __name__ == '__main__':
    # ==================== Part 1: Basic Function ====================
    A = warmUpExercise()
    print(A)

    # ======================= Part 2: Plotting =======================
    data = pd.read_csv('ex1data1.txt', header=None)
    X = data.iloc[:, 0]
    y = data.iloc[:, 1]
    m = len(y)

    # Plot Data
    # plt.plot(X, y, 'rx', ms=10)
    # plt.xlabel('Population of City in 10,000s')
    # plt.ylabel('Profit in $10,000s')
    # plt.show()

    # =================== Part 3: Cost and Gradient descent ===================
    # Add one columns at X left
    X = pd.concat([pd.Series(np.ones(m)), X], axis=1)
    theta = np.zeros((2, 1))

    # Some gradient descent settings
    iterations = 1500
    alpha = 0.01
    # compute and display initial cost
    X = X.values
    y = y.values
    J = computeCost(X, y, theta)
    print('With theta = [0 ; 0]. Cost computed = ', J)
    theta = np.array([-1, 2])
    J = computeCost(X, y, theta)
    print('With theta = [-1 ; 2]. Cost computed = ', J)

    # run gradient descent
    theta, J_history = gradientDescent(X, y, theta, alpha, iterations)
    predict1 = theta.T.dot(np.array([1, 3.5]))
    print('For population = 35,000, we predict a profit of', predict1 * 10000)
    predict2 = theta.T.dot(np.array([1, 7]))
    print('For population = 70,000, we predict a profit of', predict2 * 10000)

    plt.plot(X[:, 1], y, 'rx', ms=10, label='Training data')
    plt.plot(X[:, 1], X.dot(theta), '-', label='Linear regression')

    # ============= Part 4: Visualizing J(theta_0, theta_1) =============
    # Grid over which we will calculate J
    theta0_vals = np.linspace(-10, 10, 100)
    theta1_vals = np.linspace(-1, 4, 100)

    # initialize J_vals to a matrix of 0's
    J_vals = np.zeros((np.size(theta0_vals, 0), np.size(theta1_vals, 0)))

    for i in range(np.size(theta0_vals, 0)):
        for j in range(np.size(theta1_vals, 0)):
            t = np.array([theta0_vals[i], theta1_vals[j]])
            J_vals[i, j] = computeCost(X, y, t)

    # Surface plot
    theta0_vals, theta1_vals = np.meshgrid(theta0_vals, theta1_vals)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(theta0_vals, theta1_vals, J_vals.T)
    ax.set_xlabel(r'$\theta$0')
    ax.set_ylabel(r'$\theta$1')

    # Contour plot
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.contour(theta0_vals, theta1_vals, J_vals.T, np.logspace(-2, 3, 20))
    ax2.plot(theta[0], theta[1], 'rx', ms=10, lw=2)
    ax2.set_xlabel(r'$\theta$0')
    ax2.set_ylabel(r'$\theta$1')
    plt.show()