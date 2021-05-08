# -*- coding: utf-8 -*-
"""
Created on Sat May  8 15:34:49 2021

@author: qwuni
"""
import os
os.chdir(r'C:\Users\qwuni\OneDrive\문서\jjj.python')
print('current directory:', os.getcwd())

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
advertising=pd.read_csv('Advertising.csv', encoding='cp949', index_col=0)
len(advertising) #데이터의 개수
#train,validation,test set의 분할
train=advertising[:100]
train.head()
val= advertising[100:150]
val.head()
test=advertising[150:]
test.head()

# bestsubset selection에서 사용될 predictor들의 조합을 생성
import itertools
from copy import deepcopy
a= itertools.combinations([1,2,3,5],3)
next(a)

predictors=['TV','Radio', 'Newspaper']  ## 주의 : 파이썬은 대소문자를 구분한다!!!!
bestsubset= deepcopy(predictors)
for i in range(2, len(predictors)+1):
    bestsubset.extend([list(x) for x in list(itertools.combinations(predictors, i))])
bestsubset    

# 7개의 모형의 subset들에 대해서 모형을 적합하고
# validation set에 대한 MSE를 계산
val_mse=[]
for i in range(len(bestsubset)):
    simple_reg = sm.OLS(train['Sales'],train[bestsubset[i]]).fit()
    simple_reg_pred = np.array(simple_reg.predict(val[bestsubset[i]]))
    mse = (((simple_reg_pred-np.array(val['Sales']))**2).sum())/len(simple_reg_pred)
    val_mse.append(mse)
val_mse    

# 가장 작은 validation mse를 가지는 subset을 선택
np.argsort(val_mse)[0]  # np.argsort : 작은 값부터 순서대로 데이터의 index를 반 / 여기서 0이라는 건 가작 작은 인덱스 뜻함
min_mse_subset = bestsubset[np.argsort(val_mse)[0]]
min_mse_subset

# 앞에서 고른 bestsubset를 이용하여 test데이터에 대한 mse 계산
best_subset_reg = sm.OLS(train['Sales'], train[min_mse_subset]).fit()
best_subset_reg_pred = np.array(best_subset_reg.predict(test[min_mse_subset]))
test_mse = (((best_subset_reg_pred - np.array(test['Sales']))**2).sum())/ len(best_subset_reg_pred)
test_mse


###### 교차검증 실습
# 5-fold cross validation
k=5
# 각 validation set의 크기
val_size = int(len(advertising)/k) 
val_size
predictors = ['TV', 'Radio', 'Newspaper']

# 1개의 validation set을 분할 -> 관측치들을 번호를 분할
i = 0
idx= np.arange(len(advertising))   # 총 200개 관측치
cv_val_idx = np.arange(i*val_size, (i+1)*val_size)  # 0~39번째 관측치를 validation set 
cv_val_idx
cv_train_idx = np.array([x for x in idx if x not in cv_val_idx])  # 나머지(40~199번째)관측치를 training set
cv_train_idx

# 1개의 cross-validation set을 이용하여 mse를 계산
reg = sm.OLS(advertising['Sales'].iloc[cv_train_idx], advertising[predictors].iloc[cv_train_idx]).fit() #train set 이용
# iloc : 인덱스 행번호를  정해주면, 거기에 맞는 행데이터를 뽑아오는 pandas함수( 0부터 시작)
cv_pred = np.array(reg.predict(advertising[predictors].iloc[cv_val_idx]))
mse= (((cv_pred - np.array(advertising['Sales'].iloc[cv_val_idx]))**2).sum())/len(cv_pred)
mse

# 모든 5개의 cross validation set에 대해서 mse 계산
cv_mse=[]
for i in range(k):
    idx= np.arange(len(advertising))
    cv_val_idx= np.arange(i*val_size, (i+1)*val_size)
    cv_train_idx=np.array([x for x in idx if x not in cv_val_idx])
    reg = sm.OLS(advertising['Sales'].iloc[cv_train_idx], advertising[predictors].iloc[cv_train_idx]).fit()
    cv_pred = np.array(reg.predict(advertising[predictors].iloc[cv_val_idx]))
    mse=(((cv_pred - np.array(advertising['Sales'].iloc[cv_val_idx]))**2).sum())/len(cv_pred)
    cv_mse.append(mse)
cv_mse

np.sum(cv_mse)/k   # 5개의 cross validation mse의 평균
