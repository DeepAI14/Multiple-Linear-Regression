import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset = pd.read_csv(r'C:\workspace\MultipleLinearRegression\Multiple-Linear-Regression\House_data.csv')

x = dataset.iloc[:,np.r_[0:2,3:dataset.shape[1]]]
x = x.drop(columns=['date'])


y = dataset.iloc[:,2]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
mlr = LinearRegression()
mlr.fit(x_train,y_train)
print(mlr.coef_)
print(mlr.intercept_)
dataset.shape
x = np.append(arr=np.full((21613,1),4399739).astype(int),values=x,axis=1)
import statsmodels.api as sm
x_opt = x[:,0:20]
mlr_ols = sm.OLS(endog=y,exog=x_opt).fit()
mlr_ols.summary()
bias = mlr.score(x_train,y_train)
variance = mlr.score(x_test,y_test)
print('bias is ',bias)
print('variance is ',variance)
import statsmodels.api as sm
x_opt = x[:,np.r_[0:6,7:x.shape[1]]]
mlr_ols = sm.OLS(endog=y,exog=x_opt).fit()
mlr_ols.summary()

