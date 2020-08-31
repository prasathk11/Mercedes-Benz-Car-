#!/usr/bin/env python
# coding: utf-8

# # Importing The Required Libraries 

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


os.chdir(r"C:\Users\prasath\Music\Mercedes Benz")


# In[3]:


df_train=pd.read_csv(r"C:\Users\prasath\Music\Mercedes Benz\train.csv\train.csv")


# In[4]:


df_train.head()


# In[5]:


df_test=pd.read_csv(r"C:\Users\prasath\Music\Mercedes Benz\test.csv\test.csv")


# In[6]:


df_test.head()


# In[7]:


df_train.shape


# In[8]:


df_test.shape


# In[11]:


df_train.describe()


# In[12]:


df_test.describe()


# ### Checking Null Values in the Given Datasets

# In[13]:


df_train.isnull().sum()


# In[14]:


df_test.isnull().sum()


# In[15]:


zero_var=df_train.var()[df_train.var()==0].index.values
zero_var


# ### Removing those variable whose variance are equal to zero

# In[16]:


df_train.drop(['X11', 'X93', 'X107', 'X233', 'X235', 'X268', 'X289', 'X290',
       'X293', 'X297', 'X330', 'X347','ID'],axis=1,inplace=True)


# In[17]:


df_train.head()


# In[18]:


label_col=df_train.describe(include=['object']).columns.values
label_col


# ### Applying  label encoder.

# In[19]:


from sklearn.preprocessing import LabelEncoder


# In[20]:


le=LabelEncoder()
for col in label_col:
    le.fit(df_train[col].append(df_test[col]).values)
    df_train[col]=le.transform(df_train[col])
    df_test[col]=le.transform(df_test[col])


# ### Perform dimensionality reduction.

# In[21]:


from sklearn.decomposition import PCA
pca=PCA(0.98,svd_solver='full')


# In[22]:


X=df_train.drop(['y'],axis=1)
y=df_train['y']


# In[23]:


from sklearn.model_selection import train_test_split


# In[24]:


X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=.2,random_state=42)


# In[25]:


pca.fit(X)


# In[26]:


pca.n_components_


# In[27]:


pca.explained_variance_ratio_


# In[28]:


pca_X_train=pd.DataFrame(pca.transform(X_train))
pca_X_val=pd.DataFrame(pca.transform(X_val))


# In[29]:


print(df_train.shape)
print(df_test.shape)
print(X.shape)
print(X_train.shape)


# In[30]:


pca_X_train.head()


# In[31]:


pca_X_val.head()


# ### Predict test_df values using XGBoost.

# In[32]:


import xgboost as xgb


# In[33]:


model=xgb.XGBRegressor(objective='reg:linear',learning_rate=0.1)
model.fit(pca_X_train,y_train)


# In[34]:


pca_y_val=model.predict(pca_X_val)


# In[37]:


print("trainig accuracy")
train_acc=model.score(pca_X_train,y_train)
print(train_acc)
print("Testing Accuracy")
test_acc=model.score(pca_X_val,pca_y_val)
print(test_acc)


# # Random Forest Regressor

# In[41]:


from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor()
rf.fit(pca_X_train,y_train)
pred=rf.predict(pca_X_val)


# In[42]:


print("trainig accuracy")
train_acc=rf.score(pca_X_train,y_train)
print(train_acc)
print("Testing Accuracy")
test_acc=rf.score(pca_X_val,pred)
print(test_acc)

