
# coding: utf-8

# # feature selection

# #### Comparing models with different combinations of features

# In[1]:

# Our goal here is to check if all the available feature have to be included in the model or not.

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


# In[3]:

## Loading the csv
data = pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv',index_col = 0)

data.head()


# In[5]:

data.columns


# #### Modeling with all TV, Radio, Newspaper ad budget as independent variables

# In[21]:

## Lets build a model with all the available features
cols = ['TV', 'Radio', 'Newspaper']

X = data[cols]
Y = data.Sales

## Spliting the data into train and test
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.4,random_state = 11)

lm1 = LinearRegression()
lm1.fit(x_train,y_train)

## EValuation metrics
from sklearn import metrics
#### Modeling with TV and Radio ad budget as independent variables
print('The R square value for the model with all TV,Radio,NewsPaper Valiables is %f' %metrics.r2_score(lm1.predict(x_test),y_test))
print('The MAE square value for the model with all TV,Radio,NewsPaper Valiables is %f' %metrics.mean_absolute_error(lm1.predict(x_test),y_test))
print('The MSE square value for the model with all TV,Radio,NewsPaper Valiables is %f' %metrics.mean_squared_error(lm1.predict(x_test),y_test))
print('The RMSE square value for the model with all TV,Radio,NewsPaper Valiables is %f' %np.sqrt(metrics.mean_squared_error(lm1.predict(x_test),y_test)))


# #### Modeling with only TV ad budget as independent variables

# In[22]:

## Lets build a model with only TV independent varibale
cols = ['TV']

X = data[cols]
Y = data.Sales

## Spliting the data into train and test
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.4,random_state = 11)

lm2 = LinearRegression()
lm2.fit(x_train,y_train)

## EValuation metrics
from sklearn import metrics

print('The R square value for the model with only TV Valiables is %f' %metrics.r2_score(lm2.predict(x_test),y_test))
print('The MAE square value for the model with only TV Valiables is %f' %metrics.mean_absolute_error(lm2.predict(x_test),y_test))
print('The MSE square value for the model with only TV Valiables is %f' %metrics.mean_squared_error(lm2.predict(x_test),y_test))
print('The RMSE square value for the model with only TV Valiables is %f' %np.sqrt(metrics.mean_squared_error(lm2.predict(x_test),y_test)))


# #### Modeling with TV and Radio ad budget as independent variables

# In[25]:

## Lets build a model with only TV,Radio independent varibales
cols = ['TV','Radio']

X = data[cols]
Y = data.Sales

## Spliting the data into train and test
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.4,random_state = 11)

lm3 = LinearRegression()
lm3.fit(x_train,y_train)

## EValuation metrics
from sklearn import metrics

print('The R square value for the model with only TV,Radio Valiables is %f' %metrics.r2_score(lm3.predict(x_test),y_test))
print('The MAE square value for the model with only TV,Radio Valiables is %f' %metrics.mean_absolute_error(lm3.predict(x_test),y_test))
print('The MSE square value for the model with only TV,Radio Valiables is %f' %metrics.mean_squared_error(lm3.predict(x_test),y_test))
print('The RMSE square value for the model with only TV,Radio Valiables is %f' %np.sqrt(metrics.mean_squared_error(lm3.predict(x_test),y_test)))


# ##### From the above results we can conclude that varaible Newspaper ad budget is not significant enough in defining the sales of the client, so we remove the Newspaper varibales fromm model and will go with only TV,Radio varibales

# In[26]:

#### Instead of deriving each metric we can get the summary of metrics through the statsmodel api

import statsmodels.formula.api as smf


### Lets do the modeling with all the variables

lm4 = smf.ols(formula='Sales ~ TV+Radio+Newspaper', data=data).fit()
print(lm4.params)
print(lm4.summary())


# #### Looking at the summary above, the p value for Newspaper suggest that Newspaper could be insignificant. So can be removed form our model.

# In[27]:

### Lets model without Newspaper variable

lm5 = smf.ols(formula='Sales ~ TV+Radio', data=data).fit()
print(lm5.params)
print(lm5.summary())


# #### Fromm the above summary tables we can conclude that model without Newspaper is better, explaining more varince of the dependent varibale with less complexity. Even in terms of AIC and BIC aswel.

# #### Feature selection Using Cross validation

# In[37]:

from sklearn.cross_validation import cross_val_score

cols = ['TV','Radio','Newspaper']

X = data[cols]
Y = data.Sales

lm6 = LinearRegression()

cv_scores = cross_val_score(lm6,X,Y,cv = 10,scoring = 'mean_squared_error')
print(cv_scores)


# ##### You might be wodnering why there is -ve symbol for the mean_squared_error scores, I.e. due the fact that CV scores are used to find the best model. In classification if you remember our scoring function is accuracy, I.e higher the accuracy better is the model. So to keep the cv score criteria consistent we make the cv score for loss functions -ve.

# In[38]:

print('The RMSE with Kfold cv excluding Newspaper is %f' %np.sqrt(-cv_scores).mean())


# In[39]:

## Lets now repeat the model excluding newpaper

cols = ['TV','Radio',]

X = data[cols]
Y = data.Sales

lm7 = LinearRegression()

cv_scores = cross_val_score(lm7,X,Y,cv = 10,scoring = 'mean_squared_error')
print('The RMSE with Kfold cv excluding Newspaper is %f' %np.sqrt(-cv_scores).mean())


# In[ ]:



