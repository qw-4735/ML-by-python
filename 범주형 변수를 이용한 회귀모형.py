# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 17:14:07 2021

@author: qwuni
"""

#%%
import os
os.chdir(r'C:\Users\qwuni\OneDrive\문서\jjj.python')
print('current directory:', os.getcwd())

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

credit=pd.read_csv('Credit.csv', encoding='cp949', index_col=0)
credit = credit[['Gender', 'Balance']] # 필요한 열만을 추출
credit.head()

x = np.array(credit['Gender']) #predictor인 Gender변수선택, numpy 행렬로 변환
x[:6]
# dummy(1,-1) 변수 코딩
x_dummy = list(map(lamda x: 1 if x =='Female' else -1, x))
x_dummy[ :6]

y=np.array(credit['Balance'])  #response인 'Balance'를 numpy행렬로 변환

# 모형 적합에 사용되는 데이터생성
df = pd.DataFrame({'y' :y, 'x_dummy': x_dummy}, columns=['y','x_dummy'])
df.head()

#범주형 변수가 사용된 회귀모형 적합
categorical_reg= smf.ols(formula='y~x_dummy', data=df).fit()
categorical_reg.summary()

#%% 교호작용
advertising= pd.read_csv('advertising.csv', encoding='cp949', index_col=0)
x=np.array(advertising[['TV','radio']])

#교호작용을 나타내는 변수 추가
inter_x=x[:,0]*x[:,1]
inter_x[:6]

y=np.array(advertising['sales'])

df= pd.DataFrame({'sales':y, 'TV' :x[:,0], 'radio' :x[:,1], 'inter':inter_x}, 
                 columns= ['sales', 'TV', 'radio', 'inter'])
df.head()

inter_reg = smf.ols(formula= 'sales ~TV+radio+inter',data=df).fit()
inter_reg.summary()
 
#%% KNN
np.random.seed(1)

# train 데이터
train_x = np.sort(np.random.uniform(-1,1,200))[:, np.newaxis]
train_x.shape
train_x[:6]             
train_y = 10*train_x - 10*np.power(train_x, 3) + np.random.normal(0,1,200)[:, np.newaxis]
train_y[:6]

# test 데이터
test_x = np.random.uniform(-1,1, 100)[:, np.newaxis]
test_x[:6]
test_y = 10*test_x - 10*np.power(test_x, 3) + np.random.normal(0,1,100)[:, np.newaxis]
test_y[:6]

# train 데이터의 산점도
import matplotlib.pyplot as plt
plt.plot(train_x, train_y, linestyle= 'none', marker='o', markersize=3)

#KNN 모형을 사용하기 위한 모듈을 불러온다.
from sklearn.neighbors import KNeighborsRegressor
# k=3 인 경우의 모형을 생성
knn = KNeighborsRegressor(n_neighbors=3)
knn.fit(train_x, train_y) #training 데이터에 대해 모형 적합
predictions = knn.predict(test_x)  #test 데이터에 대한 예측 수행
predictions[:6]
mse = (((predictions - test_y) ** 2).sum()) / len(predictions)
mse

#%%
# 단순선형회귀의 적합과 MSE 계산
simple_reg = sm.OLS(train_y, train_x).fit()
simple_reg_pred = simple_reg.predict(test_x)
baseline_mse = (((simple_reg_pred - np.squeeze(test_y))**2).sum()) / len(simple_reg_pred)
baseline_mse # 단순선형회귀 mse

#여러개의 k의 값에 대해 knn 모형을 적합
mse = []
for k in range(1, 101):
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(train_x, train_y)
    knn_pred= knn.predict(test_x)
    mse.append((((knn_pred - test_y) **2).sum()) / len(knn_pred))
mse

# test mse의 시각화
import matplotlib.pyplot as plt
plt.hlines(baseline_mse, 0, 1)
plt.plot(1/np.arange(1,101), mse)


