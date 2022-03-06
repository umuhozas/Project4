# Project4
This repository contains a multiple boosting algorithm and its application on combinations of different regressors. In this project I am going to use the concrete dataset. The original dataset has nine independent variables, but I will be using three independent variables('cement','water', and 'superplastic') to predict the strength of the cement.

To begin our analysis, I am going to start by importing someimportant libraries that I will need.

```Python

import numpy as np
import pandas as pd
from scipy.linalg import lstsq
from scipy.sparse.linalg import lsmr
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, griddata, LinearNDInterpolator, NearestNDInterpolator
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, train_test_split as tts
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
import matplotlib.pyplot as plt
from matplotlib import pyplot
```

For this analysis, I am not going to use stat models or kernel regression, so I will go ahead and define our kernel functions instead.

```Python
# Tricubic Kernel
def Tricubic(x):
  if len(x.shape) == 1:
    x = x.reshape(-1,1)
  d = np.sqrt(np.sum(x**2,axis=1))
  return np.where(d>1,0,70/81*(1-d**3)**3)

# Quartic Kernel
def Quartic(x):
  if len(x.shape) == 1:
    x = x.reshape(-1,1)
  d = np.sqrt(np.sum(x**2,axis=1))
  return np.where(d>1,0,15/16*(1-d**2)**2)

# Epanechnikov Kernel
def Epanechnikov(x):
  if len(x.shape) == 1:
    x = x.reshape(-1,1)
  d = np.sqrt(np.sum(x**2,axis=1))
  return np.where(d>1,0,3/4*(1-d**2)) 
```

After defining our kernel functions, I am going to define a kernel local regression model which will take different parameters like dependent variable, independent variable, our choice of kernel, intercept, and others.

```Python
#Defining the kernel local regression model

def lw_reg(X, y, xnew, kern, tau, intercept):
    # tau is called bandwidth K((x-x[i])/(2*tau))
    n = len(X) # the number of observations
    yest = np.zeros(n)

    if len(y.shape)==1: # here we make column vectors
      y = y.reshape(-1,1)

    if len(X.shape)==1:
      X = X.reshape(-1,1)
    
    if intercept:
      X1 = np.column_stack([np.ones((len(X),1)),X])
    else:
      X1 = X

    w = np.array([kern((X - X[i])/(2*tau)) for i in range(n)]) # here we compute n vectors of weights

    #Looping through all X-points
    for i in range(n):          
        W = np.diag(w[:,i])
        b = np.transpose(X1).dot(W).dot(y)
        A = np.transpose(X1).dot(W).dot(X1)
        beta, res, rnk, s = lstsq(A, b)
        yest[i] = np.dot(X1[i],beta)
    if X.shape[1]==1:
      f = interp1d(X.flatten(),yest,fill_value='extrapolate')
    else:
      f = LinearNDInterpolator(X, yest)
    output = f(xnew) # the output may have NaN's where the data points from xnew are outside the convex hull of X
    if sum(np.isnan(output))>0:
      g = NearestNDInterpolator(X,y.ravel()) 
      output[np.isnan(output)] = g(xnew[np.isnan(output)])
    return output
```
After our local regression model, I am going to define function to boost our current local regression model. For training oue boosted method, I will use X and y that I will define below from our cement dataset.

```Python
def boosted_lwr(X, y, xnew, kern, tau, intercept, model_boosting, nboost):
  # we need decision trees
  # for training the boosted method we use X and y
  Fx = lw_reg(X,y,X,kern,tau,intercept) # we need this for training the Decision Tree
  output = booster(X,y,xnew,kern,tau,model_boosting,nboost)
  return output 
```
After boosting our local regression model, I am going to write one more function which will specify the chosen regression model to boost, the number of times or function will repeat the boosting, choice of kernel, tau, etc

```Python
def booster(X,y,xnew,kern,tau,model_boosting,nboost):
  Fx = lw_reg(X,y,X,kern,tau,True)
  Fx_new = lw_reg(X,y,xnew,kern,tau,True)
  new_y = y - Fx
  output = Fx
  output_new = Fx_new
  for i in range(nboost):
    model_boosting.fit(X,new_y)
    output += model_boosting.predict(X)
    output_new += model_boosting.predict(xnew)
    new_y = y - output
  return output_new
```
Let's import our data

```Python
concrete = pd.read_csv('drive/MyDrive/Colab Notebooks/Data_410/data/concrete.csv')

```
Let's define our dependent and independent variables

```Python
X = concrete[['cement','water','superplastic']].values
y = concrete['strength'].values
```
For our model boosting we choose from all regression models, but I am going to use Random Forest Regressor with one hundred trees and max_depth = 3. After this I will also standardize my data using StandardScaler.

```Python
model_boosting = RandomForestRegressor(n_estimators=100,max_depth=3)
scale = StandardScaler() 
xscaled = scale.fit_transform(X)
```
Let's split our data using KFold cross validation
```Python
for i in [1234]:
  kf = KFold(n_splits=10,shuffle=True,random_state=i)
  # this is the Cross-Validation Loop
  for idxtrain, idxtest in kf.split(X):
    xtrain = X[idxtrain]
    ytrain = y[idxtrain]
    ytest = y[idxtest]
    xtest = X[idxtest]
    xtrain = scale.fit_transform(xtrain)
    xtest = scale.transform(xtest)
```
Let's repeat our boosting twice and see what we get
```Python
yhat = boosted_lwr(xtrain,ytrain,xtest,Tricubic,1,True,model_boosting,2)
mse(ytest, yhat)

```
mse (ytest, yhat) = 153.18129890828084
This MSE is pretty big, I am going to try gradient boosting on our data and see if it makes an improvement.

```Python
import xgboost as xgb
```

Let's tru a nested cross-validation for better comparison

```Python
# we want more nested cross-validations

mse_blwr = []
mse_xgb = []


for i in [123]:
  kf = KFold(n_splits=10,shuffle=True,random_state=i)
  # this is the Cross-Validation Loop
  for idxtrain, idxtest in kf.split(X):
    xtrain = X[idxtrain]
    ytrain = y[idxtrain]
    ytest = y[idxtest]
    xtest = X[idxtest]
    xtrain = scale.fit_transform(xtrain)
    xtest = scale.transform(xtest)
    dat_train =np.concatenate([xtrain,ytrain.reshape(-1,1)], axis =1)
    dat_test= np.concatenate([xtest, ytest.reshape(-1,1)], axis = 1)
    yhat_blwr = boosted_lwr(xtrain,ytrain, xtest,Tricubic,1,True, model_boosting,2)
    model_xgb = xgb.XGBRegressor(objective ='reg:squarederror',n_estimators=100,reg_lambda=20,alpha=1,gamma=10,max_depth=3)
    model_xgb.fit(xtrain,ytrain)
    yhat_xgb = model_xgb.predict(xtest)
    mse_blwr.append(mse(ytest,yhat_blwr))
    mse_xgb.append(mse(ytest,yhat_xgb))
print('The Cross-validated Mean Squared Error for BLWR is : '+str(np.mean(mse_blwr)))
print('The Cross-validated Mean Squared Error for XGB is : '+str(np.mean(mse_xgb)))
```
The Cross-validated Mean Squared Error for BLWR is : 150.67848506482736
The Cross-validated Mean Squared Error for XGB is : 151.33740690420456

Our MSE for Boosted locally weighted regression improved, but the MSE for gradient boosting is not better.

##LightGBM 

```Python

```

```Python

```

```Python


