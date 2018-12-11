
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df=pd.read_csv('train.csv')


# **1 Importing libs and Data**

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew
import os


# **2 loading the data**

# In[5]:


#loading the data
c_train = pd.read_csv('train.csv')
c_test = pd.read_csv('test.csv')


Price = c_train['SalePrice']
c_train = c_train.drop('SalePrice',axis = 1)
c_train.drop('Id',axis = 1,inplace = True)
print(c_train.shape)


# **3 analysis the data**
# 
# 
# 
# 
# 

# In[6]:


#frist lets give a look at the house price data
print(Price.describe())
#check whether it is norm
sns.distplot(Price ,hist = True, fit=norm,color = 'red')
(mu, sigma) = norm.fit(Price)
plt.ylabel('number')
plt.xlabel('Price')
plt.legend(['Normal. ($\mu=$ {:.1f} and $\sigma=$ {:.1f} )'.format(mu, sigma)],loc='best')
#there is no problem it is right skewed


# In[14]:


#then trying to find the correlation for all features
corr_train = c_train.corr()
plt.subplots(figsize=(10,10))
sns.heatmap(corr_train,square=True, cmap="YlGnBu")


# In[15]:


log_Price = np.log(Price)
sns.distplot(log_Price , fit=norm,hist = True,color = 'blue');
(mu, sigma) = norm.fit(log_Price )
plt.legend(['Normal ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],loc='best')
plt.title('Price (Log)');


# In[16]:


b = Price.skew()
a = log_Price.skew()
print("skewness before: " + str(b))
print("skewness now: " + str(a))


# **4 data processing** 
# 
# 
# 
# 

# In[17]:


num_na = (c_train.isnull().mean())
ratio_na = num_na.sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing_Ratio' :ratio_na})
print(missing_data.head(10))


# In[18]:


c_train.drop('PoolQC', axis=1, inplace=True)
c_train.drop('MiscFeature', axis=1, inplace=True)
c_train.drop('Alley', axis=1, inplace=True)
c_train.drop('Fence', axis=1, inplace=True)
c_train.drop('FireplaceQu', axis=1, inplace=True)


# In[19]:





data = c_train
null_columns = data.columns[data.isnull().any()]
data[null_columns].isnull().sum()


data["LotFrontage"] = data["LotFrontage"].fillna(np.mean(data["LotFrontage"]))
data["MasVnrType"] = data["MasVnrType"].fillna("None")
data["MasVnrArea"] = data["MasVnrArea"].fillna(np.mean(data["MasVnrArea"]))
data["BsmtQual"] = data["BsmtQual"].fillna("None")
data["BsmtCond"] = data["BsmtCond"].fillna("None")
data["BsmtExposure"] = data["BsmtExposure"].fillna("None")
data["BsmtFinType1"] = data["BsmtFinType1"].fillna("None")
data["BsmtFinType2"] = data["BsmtFinType2"].fillna("None")
data["Electrical"] = data["Electrical"].fillna("None")
data["GarageType"] = data["GarageType"].fillna("None")
data["GarageYrBlt"] = data["GarageYrBlt"].fillna(np.mean(data["GarageYrBlt"]))
data["GarageFinish"] = data["GarageFinish"].fillna("None")
data["GarageQual"] = data["GarageQual"].fillna("None")
data["GarageCond"] = data["GarageCond"].fillna("None")

null_columns = data.columns[data.isnull().any()]
data[null_columns].isnull().sum()

#transfer into one hot coding 
one_hot_train = pd.get_dummies(data)

print(data.shape)
print(one_hot_train.shape)


# **5 Prediction via Random forest**
# 
# 
# 

# In[20]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, make_scorer, mean_squared_error


# In[21]:


#splite the data into test and train sets
X_train,X_test, y_train, y_test = train_test_split(one_hot_train, log_Price, random_state=1)
rf_clf = RandomForestRegressor(random_state = 1)
tuned_para =  [{'n_estimators': [60,120,500],'max_features': ['auto', 'sqrt'],'min_samples_leaf': [1,3,5],'max_depth': [ 5, 15,25]}]
grid_clf = GridSearchCV(rf_clf,param_grid=tuned_para,n_jobs=-1,cv=5)
grid_fit = grid_clf.fit(X_train,y_train)
mu = grid_clf.cv_results_['mean_test_score']
stds = grid_clf.cv_results_['std_test_score']
print(mu)


# In[23]:


#print the best parameter
rf_opt = grid_fit.best_estimator_
print(grid_fit.best_params_)
#print(grid_fit.best_estimator_)

#find the 10 most important params
importances = rf_opt.feature_importances_
print(X_train.columns.values[(np.argsort(importances)[::-1])[:10]])


# In[24]:


#test the performance
rf_opt_clf = RandomForestRegressor(random_state=1, max_features='auto', n_estimators= 500, max_depth=15,min_samples_leaf = 1 )
rf_opt_clf.fit(X_train,y_train)
rf_opt_pred = rf_opt_clf.predict(X_test)
y_l = y_test.tolist()
rf = rf_opt_pred.tolist()
L1_norm = abs(y_l-rf_opt_pred).mean()
rf_R2 = r2_score(y_test,rf_opt_pred)
rf_mse = mean_squared_error(y_test, rf_opt_pred)


# In[25]:


print(L1_norm)


# In[26]:


d = {'RF': [rf_R2, rf_mse,L1_norm]}
d_i = ['R2', 'Mean Squared Error','L1_norm']
df_results = pd.DataFrame(data=d, index = d_i)
df_results

