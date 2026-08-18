import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset = pd.read_csv(r'C:\workspace\MultipleLinearRegression\Investment.csv')
x = dataset.iloc[:,:-1]
y = dataset.iloc[:,4]
x = pd.get_dummies(x,dtype=int)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(regressor.coef_)
print(regressor.intercept_)
#x = np.append(arr=np.full((50,1),42467).astype(int),values=x,axis=1)

import statsmodels.api as sm
def backward_elimination(x_data,y_data,sig_level = 0.05):
    x_opt = x_data.copy()
    while(x_opt.shape[1]>0):
        model = sm.OLS(y_data,sm.add_constant(x_opt)).fit()
        pvalues = model.pvalues.drop('const')
        worst_p = pvalues.max()
        if(worst_p > sig_level):
            feature_drop = pvalues.idxmax()
            x_opt= x_opt.drop(columns=[feature_drop])
        else:
            break
        
    return x_opt,model    
   
x_train_opt,final_model=  backward_elimination(x_train,y_train,sig_level = 0.05)
print(final_model.summary())
selected_columns = x_train_opt.columns
x_test_opt = x_test[selected_columns]
regressor_final = LinearRegression()
regressor_final.fit(x_train_opt,y_train)
bias_final = regressor_final.score(x_train_opt,y_train)
variance_final = regressor_final.score(x_test_opt,y_test)
print(bias_final)
print(variance_final)

#import statsmodels.api as sm
#x_opt = x[:,[0,1,2,3,4,5]]
#regressor_ols = sm.OLS(endog=y,exog=x_opt).fit()
#regressor_ols.summary()


#regressor_ols = sm.OLS(endog=y,exog=x_opt).fit()
#regressor_ols.summary()

