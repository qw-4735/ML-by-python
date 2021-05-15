# -*- coding: utf-8 -*-
"""
Created on Mon May 10 16:27:08 2021

@author: qwuni
"""
## Poisson Regression with Gradient Descent

import numpy as np
np.random.seed(1)
p=2
true_beta = np.array([[1],[0.5]])
n = 1000
x = np.random.normal(loc=0, scale=0.2, size=(n,1))
x= np.hstack((np.ones((n,1)),x))
x[:5, :]
# 모수값을 계산하고 poisson 분포를 이용해 y를 생성
parm = np.exp(x @ true_beta)
parm[:5, :]
y = np.random.poisson(parm)
y[:5, :]

beta = np.array([.5, .5]).reshape((p,1))  # 베타 초기값
# 1차미분
parm = np.exp(x @ beta)
grad = -np.mean(y*x - parm*x, axis=0).reshape((p,1))  # gradient vector 계산
grad

learning_rate = 0.7
# initial beta
beta = np.zeros((p,1))
for i in range(500):
    # 모수추정값, 1차 미분 계산
    parm = np.exp(x @ beta)
    grad = -np.mean(y*x - parm*x, axis=0).reshape((p,1))
    # beta update
    beta_new = beta - learning_rate*grad
    # stopping rule
    if np.sum(np.abs(beta_new - beta)) < 1e-8:
        beta = beta_new
        print('Iteration {} beta:'.format(i+1))
        print(beta, '\n')
        break
    else:
        beta = beta_new
        print('Iteration {} beta:'.format(i+1))
        print(beta, '\n')
        
##### Poisson Regression with Newton-Raphson Method
beta = np.array([.5,.5]).reshape((p,1))
# 2차 미분
parm = np.exp(x @ beta)
D = np.diag(np.squeeze(parm))  # np.diag : 대각행렬을 만들어라( 대각원소만 뽑아낸다)
D
H = x.T @ D @ x/n  # .T : 행렬의 transpose   / @ : 행렬곱
H

# initial beta
beta = np.zeros((p,1))
for i in range(500):
    # 모수추정값, 1차와 2차 미분 계산
    parm = np.exp(x @ beta)
    grad= -np.mean(y*x - parm*x, axis=0).reshape((p,1))
    D = np.diag(np.squeeze(parm))
    H = x.T @ D @ x/n
    # beta update
    beta_new = beta - np.linalg.inv(H) @ grad
    # stopping rule
    if np.sum(np.abs(beta_new - beta)) < 1e-8:
        beta= beta_new
        print('Iteration {} beta:'.format(i+1))
        print(beta, '\n')
        break
    else:
        beta= beta_new
        print('Iteration {} beta:'.format(i+1))
        print(beta, '\n')
        break 




















        























