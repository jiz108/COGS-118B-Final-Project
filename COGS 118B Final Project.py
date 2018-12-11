#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import urllib
import sklearn
import scipy.optimize
import random
import math
import ast
import numpy
from numpy import dot
from numpy.linalg import norm
from collections import Counter
from urllib.request import urlopen
from math import exp
from math import log
import gzip
from collections import defaultdict
import nltk
import string
from nltk.stem.porter import *
from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np
from scipy import spatial
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# In[22]:


#Read data
rawdata = pd.read_csv("train.csv") 


# In[23]:


#Pre-processing data, get rid of N/A
rawdata["LotFrontage"] = rawdata["LotFrontage"].fillna(np.mean(rawdata["LotFrontage"]))
rawdata["Alley"] = rawdata["Alley"].fillna("None")
rawdata["MasVnrType"] = rawdata["MasVnrType"].fillna("None")
rawdata["MasVnrArea"] = rawdata["MasVnrArea"].fillna(np.mean(rawdata["MasVnrArea"]))
rawdata["BsmtQual"] = rawdata["BsmtQual"].fillna("None")
rawdata["BsmtCond"] = rawdata["BsmtCond"].fillna("None")
rawdata["BsmtExposure"] = rawdata["BsmtExposure"].fillna("None")
rawdata["BsmtFinType1"] = rawdata["BsmtFinType1"].fillna("None")
rawdata["BsmtFinType2"] = rawdata["BsmtFinType2"].fillna("None")
rawdata["Electrical"] = rawdata["Electrical"].fillna("None")
rawdata["FireplaceQu"] = rawdata["FireplaceQu"].fillna("None")
rawdata["GarageType"] = rawdata["GarageType"].fillna("None")
rawdata["GarageYrBlt"] = rawdata["GarageYrBlt"].fillna(np.mean(rawdata["GarageYrBlt"]))
rawdata["GarageFinish"] = rawdata["GarageFinish"].fillna("None")
rawdata["GarageQual"] = rawdata["GarageQual"].fillna("None")
rawdata["GarageCond"] = rawdata["GarageCond"].fillna("None")
rawdata["PoolQC"] = rawdata["PoolQC"].fillna("None")
rawdata["Fence"] = rawdata["Fence"].fillna("None")
rawdata["MiscFeature"] = rawdata["MiscFeature"].fillna("None")


# In[24]:


#Build dict for houses in train and test
sale_price_dict = defaultdict(float)
for i in range(len(rawdata)):
    l = rawdata.loc[i]['Id']
    s = rawdata.loc[i]['SalePrice']
    sale_price_dict[l] = s

#Make it to 1/0 
rawdata = pd.get_dummies(rawdata)

#Drop 'SalePrice'
rawdata = rawdata.drop('SalePrice',1)

#Splite to train and test
train, test = train_test_split(rawdata,test_size = 0.2)
train = train.reset_index()
test = test.reset_index()


# In[5]:


#Check 'test.csv' contain N/A or not
#null_columns = test.columns[test.isnull().any()]
#test[null_columns].isnull().sum()


# In[6]:


#Check 'train.csv' contain N/A or not
null_columns = rawdata.columns[rawdata.isnull().any()]
rawdata[null_columns].isnull().sum()


# In[9]:


#data visualization
X = []
Y = []
data = []
sale_price_dict = defaultdict(int)

for i in range(len(rawdata)):
    data.append(rawdata.loc[i])

for i in data:
    sale_price_dict[i['SalePrice']] += 1
    
for i in sale_price_dict:
    X.append(i)
    Y.append(sale_price_dict[i])

fig = plt.figure(figsize=(20,10))
plt.scatter(X, Y, marker='o', s=50)
plt.ylabel("Number of houses")
plt.xlabel("Price")
plt.grid()


# In[25]:


#cosine similarity
def cos_sim(house1, house2):
    cos_sim = dot(house1, house2)/(norm(house1)*norm(house2))
    return cos_sim


# In[74]:


#predict function
def predict(house,train_set):

    #convert to list
    house = house.tolist()
    simList = []

    for i in range(len(train_set)):
        house2 = train_set.loc[i]
        
        #Get ID and Price 
        Id = house2['Id']
        price = sale_price_dict[Id]

        #Make to list for calculate similarity 
        house2 = house2.tolist()
        cos = cos_sim(house,house2)
        simList.append((cos,Id,price))
        
        
    simList.sort()
    simList.reverse()
    mostCommon = [x[2] for x in simList[:15]]
    
    #Calculate the average money
    total = 0
    for i in mostCommon:
        total += i
    return float(total/len(mostCommon))


# In[75]:


#Test
predicts = []
answer = []
for i in range(len(test)):
    house = test.loc[i]
    predicts.append(predict(house,train))
    answer.append(sale_price_dict[test.loc[i]['Id']])


# In[76]:


#Average error
total = 0
for i in range(len(answer)):
    p = predicts[i]
    a = answer[i]
    acc =1 -  (float(abs(p-a)/a))
    total += acc
print('Accuracy: ')
print(float(total/len(answer)))


# In[77]:


from sklearn.metrics import r2_score
r2_score(answer, predicts)


# In[ ]:




