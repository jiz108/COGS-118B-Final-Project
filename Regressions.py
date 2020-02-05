
# coding: utf-8

# In[6]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
import matplotlib
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# In[7]:


data = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")


# In[8]:


data.shape


# In[9]:


data.head()


# In[10]:


data.columns


# In[11]:


len(set(data["Neighborhood"]))


# In[12]:


plt.scatter(data["GrLivArea"], data["SalePrice"])
plt.xlabel("GrLivArea")
plt.ylabel("SalePrice")
plt.grid(True)
plt.savefig("saleprice")
plt.show()


# In[13]:


data = data.drop(data[(data['GrLivArea']>4000) & (data['SalePrice']<200000)].index)
data = data.drop(columns = ["Id"])


# In[14]:


sns.heatmap(data.corr())


# In[15]:


null_columns = data.columns[data.isnull().any()]
data[null_columns].isnull().sum()


# In[16]:


data["LotFrontage"] = data["LotFrontage"].fillna(np.mean(data["LotFrontage"]))
data["Alley"] = data["Alley"].fillna("None")
data["MasVnrType"] = data["MasVnrType"].fillna("None")
data["MasVnrArea"] = data["MasVnrArea"].fillna(np.mean(data["MasVnrArea"]))
data["BsmtQual"] = data["BsmtQual"].fillna("None")
data["BsmtCond"] = data["BsmtCond"].fillna("None")
data["BsmtExposure"] = data["BsmtExposure"].fillna("None")
data["BsmtFinType1"] = data["BsmtFinType1"].fillna("None")
data["BsmtFinType2"] = data["BsmtFinType2"].fillna("None")
data["Electrical"] = data["Electrical"].fillna("None")
data["FireplaceQu"] = data["FireplaceQu"].fillna("None")
data["GarageType"] = data["GarageType"].fillna("None")
data["GarageYrBlt"] = data["GarageYrBlt"].fillna(np.mean(data["GarageYrBlt"]))
data["GarageFinish"] = data["GarageFinish"].fillna("None")
data["GarageQual"] = data["GarageQual"].fillna("None")
data["GarageCond"] = data["GarageCond"].fillna("None")
data["PoolQC"] = data["PoolQC"].fillna("None")
data["Fence"] = data["Fence"].fillna("None")
data["MiscFeature"] = data["MiscFeature"].fillna("None")


# In[17]:


null_columns = data.columns[data.isnull().any()]
data[null_columns].isnull().sum()


# In[18]:


onehot = pd.get_dummies(data)


# In[19]:


onehot.to_csv("onehot.csv")


# In[20]:


onehot.shape


# In[23]:


X_train = onehot[:int(0.8 * data.shape[0])].drop(columns = ["SalePrice"])
X_test = onehot[int(0.8 * data.shape[0]):].drop(columns = ["SalePrice"])
y_train = onehot["SalePrice"][:int(0.8 * data.shape[0])]
y_test = onehot["SalePrice"][int(0.8 * data.shape[0]):]


# In[22]:


y_train = np.log1p(y_train)
y_test = np.log1p(y_test)


# In[26]:


clf2 = Ridge()
alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75, 100, 150]
parameters = {'alpha':alphas}
ridge = GridSearchCV(clf2, parameters, cv = 5, return_train_score=True)
ridge.fit(X_train, y_train)
alpha_star2 = ridge.best_params_


# In[27]:


alpha_star2


# In[28]:


ridge.cv_results_ 


# In[29]:


plt.plot(alphas, list(ridge.cv_results_["mean_test_score"]))
plt.ylabel("Score")
plt.xlabel("Regularization Term")
plt.title("Score curve")
plt.grid(True)
plt.savefig("ridge.png")


# In[32]:


ridge_best = Ridge(alpha = alpha_star2['alpha'])
ridge_best.fit(X_train, y_train)


# In[33]:


mean_squared_error(ridge_best.predict(X_test), y_test)


# In[34]:


r2_score(ridge_best.predict(X_test), y_test)


# In[35]:


clf1 = linear_model.Lasso(tol = 0.027)
alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75, 100, 150]
parameters = {'alpha':alphas}
lasso = GridSearchCV(clf1, parameters, cv = 5)
lasso.fit(X_train, y_train)
alpha_star1 = lasso.best_params_


# In[36]:


alpha_star1


# In[37]:


plt.plot(alphas, list(lasso.cv_results_["mean_test_score"]))
plt.ylabel("Score")
plt.xlabel("Regularization Parameter")
plt.grid(True)
plt.title("Score Curve")
plt.savefig("lasso.png")


# In[38]:


lasso_best = linear_model.Lasso(alpha = alpha_star1['alpha'])
lasso_best.fit(X_train, y_train)


# In[39]:


mean_squared_error(lasso_best.predict(X_test), y_test)


# In[41]:


r2_score(lasso_best.predict(X_test), y_test)


# In[401]:


coef = pd.Series(lasso_best.coef_, index = X_train.columns)


# In[402]:


print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")


# In[403]:


imp_coef = pd.concat([coef.sort_values().head(10),
                     coef.sort_values().tail(10)])


# In[404]:


matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Coefficients in the Lasso Model")
plt.show()
plt.savefig("coef_lasso.png")


# In[405]:


matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)

preds = pd.DataFrame({"preds":model_lasso.predict(X_train), "true":y_train})
preds["residuals"] = preds["true"] - preds["preds"]
preds.plot(x = "preds", y = "residuals",kind = "scatter")


# In[453]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib

import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats.stats import pearsonr


get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina' #set 'png' here when working on notebook")
get_ipython().run_line_magic('matplotlib', 'inline')


# In[454]:


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")


# In[455]:


all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],
                      test.loc[:,'MSSubClass':'SaleCondition']))


# In[456]:


matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)
prices = pd.DataFrame({"price":train["SalePrice"], "log(price + 1)":np.log1p(train["SalePrice"])})
prices.hist()


# In[457]:


#log transform the target:
train["SalePrice"] = np.log1p(train["SalePrice"])

#log transform skewed numeric features:
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index

all_data[skewed_feats] = np.log1p(all_data[skewed_feats])


# In[458]:


all_data = pd.get_dummies(all_data)


# In[459]:


#filling NA's with the mean of the column:
all_data = all_data.fillna(all_data.mean())


# In[460]:


#creating matrices for sklearn:
X_train = all_data[:train.shape[0]]
X_test = all_data[train.shape[0]:]
y = train.SalePrice


# In[461]:


from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score

def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)


# In[462]:


model_ridge = Ridge()


# In[463]:


alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() 
            for alpha in alphas]


# In[464]:


cv_ridge = pd.Series(cv_ridge, index = alphas)
cv_ridge.plot(title = "Validation - Just Do It")
plt.xlabel("alpha")
plt.ylabel("rmse")


# In[465]:


cv_ridge.min()


# In[466]:


model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(X_train, y)


# In[467]:


rmse_cv(model_lasso).mean()


# In[468]:


coef = pd.Series(model_lasso.coef_, index = X_train.columns)


# In[469]:


print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")


# In[470]:


imp_coef = pd.concat([coef.sort_values().head(10),
                     coef.sort_values().tail(10)])


# In[474]:


matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Coefficients in the Lasso Model")
plt.savefig("feature_selection.png")


# In[481]:


#let's look at the residuals as well:
matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)

preds = pd.DataFrame({"preds":model_lasso.predict(X_train), "true":y})
preds["residuals"] = preds["true"] - preds["preds"]
preds.plot(x = "preds", y = "residuals",kind = "scatter")

