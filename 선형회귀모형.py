# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#%%
import os
os.chdir(r'C:\Users\qwuni\OneDrive\문서\jjj.python')
print('current directory:', os.getcwd())

#%% mtcars

#pandas 모듈을 불러온다.
import pandas as pd

#csv 파일 형식으로 되어있는 mtcars 데이터를 불러온다.
mtcars= pd.read_csv('mtcars.csv',encoding='cp949')

print(mtcars.head() )

#%%
import numpy as np
import statsmodels.api as sm
N=1000   #관측치의 개수
X=np.random.normal(loc=0, scale=1, size=(N,1)) #정규분포를 이용해 데이터를 생성(N(0,1))
X=sm.add_constant(X) #절편(상수항)추가
X[ :6, :]
eplison = np.random.normal(loc=0, scale=1, size=(N,1)) #정규분포를 이용해 오차항 생성 N(0,1)
# 실제 모형: Y= 2 +3X + epsilon
Y= X @ np.array([[2],[3]])+eplison
Y[ :6, :]

simple_reg = sm.OLS(Y,X).fit() #단순회귀모형 적합
simple_reg.summary()

#%%
N=1000 
repeat_num= 1000 #모형 적합을 반복하는 횟수
beta = np.zeros((2,))
for k in range(repeat_num):
    X=np.random.normal(loc=0, scale=1, size=(N,1))
    X=sm.add_constant(X)
    epsilon = np.random.normal(loc=0, scale=1, size=(N,1))
    Y= X @ np.array([[2],[3]])+ epsilon
    
    simple_reg = sm.OLS(Y,X).fit()
    beta += simple_reg.params
beta /= repeat_num    #적합된 회귀계수의 평균을 계산
beta

#%% 단순회귀모형(광고 매출 분석)
import pandas as pd
advertising= pd.read_csv('advertising.csv', encoding='cp949', index_col=0)
X= np.array(advertising[['TV']]) # TV변수를 선택, numpy 행렬로 변환
X= sm.add_constant(X)
X[ :6, :]
Y = np.array(advertising[['sales']]) # 반응변수인 'sales'를 numpy 행렬로 변환
Y[ :6, :]

simple_reg = sm.OLS(Y,X).fit()
simple_reg.summary()

#%% 다변량회귀분석
import pandas as pd
advertising= pd.read_csv('advertising.csv', encoding='cp949', index_col=0)
X= np.array(advertising[['TV', 'radio', 'newsapaer']]) #모든 설명변수를 선택, numpy 행렬로 변환
X= sm.add_constant(X)
X[ :6, :]
Y= np.array(advertising[['sales']])
Y[ :6, :]

multi_reg = sm.OLS(Y,X).fit() #다변량회귀모형 적합
multi_reg.summary()

#%% 모형평가(bic 활용)
N=1000
X= np.random.normal(loc=0, scale=1, size=(N,4))
x[ :6, :]
beta= np.array([[0],[3],[1],[0]]) #실제 회귀계수
beta
epsilon = np.random.normal(loc=0, scale=1, size=(N,1))
Y= X @ beta + epsilon  #참모형 : Y = 0*X1 + 3*X2 + 1*X3 + 0*X4 +epsilon
Y[ :6, :]

candidates = list(range(4))  #평가할 후보 변수
candidates

# step1
bic = []
for predictor in candidates:
    reg = sm.OLS(Y,X[:,predictor]).fit() #회귀모형 적합
    bic.append(reg.bic) #bic 값 저장
bic 

chosen1 = np.argmin(bic) #bic 값이 가장 작은 변수를 선택
chosen1

del candidates[chosen1] #선택한 변수를 후보에서 제외
candidates

# step2
bic= []
for predictor in candidates:
    reg= sm.OLS(Y,X[:, [chosen1, predictor]]).fit() #선택한 설명변수를 포함해 회귀모형 적합
    bic.append(reg.bic)
bic 

chosen2 = np.argmin(bic)
chosen2

del candidates[chosen2]
chosen2

# step3
bic= []
for predictor in candidates :
    reg = sm.OLS( Y,X[:, [chosen1, chosen2, predictor]]).fit()
    bic.append(reg.bic)
bic 

chosen3= np.argmin(bic)
chosen3


