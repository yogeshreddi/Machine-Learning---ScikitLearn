
# coding: utf-8

# # Classification

# ### KNN - K Nearest Neigbours

# In[1]:

# We will use Iris data available in scikit learn for our model 

from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier


# In[13]:

# Loading data

iris = load_iris()
X = iris.data
Y = iris.target


# Instantiating estimators, I.e. select the model you want to use... KNN 

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X,Y)


new_data = [[3,5,4,2],[5,3,1,0]]
print(knn.predict(new_data))


# ### Logistic Regression

# In[14]:

# Lets try Logestic on the same dataset

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(X,Y)
print(logreg.predict(new_data))


# ### Decision Trees

# In[16]:

# Lets try decsion tree clasifier

from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier()
DT.fit(X,Y)
print(DT.predict(new_data))


# In[17]:

# Lets try Extra trees

from sklearn.tree import ExtraTreeClassifier
ET = DecisionTreeClassifier()
ET.fit(X,Y)
print(ET.predict(new_data))


# In[19]:

print(iris.target_names)


# ### Linear Regression 

# In[24]:


import pandas as pd
import numpy as np

# Reading data from csv
data = pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv',index_col=0)


# In[25]:

#to look at the first five rows of the object data

data.head()


# In[26]:

#to look at the last five rows of the object data

data.tail()


# In[23]:

# to look at the number of columns and observations in the data

data.shape


# In[29]:

## Seaborn for visulization
import seaborn as sns

get_ipython().magic('matplotlib inline')


# In[34]:

sns.pairplot(data, x_vars=['TV','Radio','Newspaper'],y_vars='Sales',size = 7, aspect=0.8,kind = 'reg')


# #### Building the LR model on complete data and tetsing on the same data

# In[27]:

from sklearn.linear_model import LinearRegression
from sklearn import metrics


# In[59]:

cols = ['TV','Radio','Newspaper']

X = data[cols]
Y = data['Sales']

lr = LinearRegression()
lr.fit(X,Y)


# In[60]:

print('R square value for lr on complete on data is %f' %metrics.r2_score(lr.predict(X),Y))
print ('INtercept is %f'%lr.intercept_)
print (lr.coef_)
print(zip(cols,lr.coef_))


# #### Lets now use train test splitting to test our accuracy

# In[64]:

from sklearn.cross_validation import train_test_split

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.3,random_state = 5)


# In[66]:

X = data[cols]
Y = data[['Sales']]

lr = LinearRegression()
lr.fit(x_train,y_train)

print('R square value for lr on complete on training is %f' %metrics.r2_score(lr.predict(x_train),y_train))
print('R square value for lr on complete on testing is %f' %metrics.r2_score(lr.predict(x_test),y_test))


# In[67]:

print ('INtercept is %f'%lr.intercept_)
print (lr.coef_)
print(zip(cols,lr.coef_))


# #### EValuation metrcis for linear regression

# In[76]:

print('R square value for lr on complete on testing is %f' %metrics.r2_score(lr.predict(x_test),y_test))
print('MAE value for lr on complete on testing is %f' %metrics.mean_absolute_error(lr.predict(x_test),y_test))
print('MSE value for lr on complete on testing is %f' %metrics.mean_squared_error(lr.predict(x_test),y_test))
print('RMSE value for lr on complete on testing is %f' %np.sqrt(metrics.mean_squared_error(lr.predict(x_test),y_test)))


# In[80]:

###Using STATSMODELS ####

import statsmodels.formula.api as smf

lm = smf.ols(formula='Sales ~ TV+Radio+Newspaper', data=data).fit()

lm.params


# In[81]:

lm.summary()

