#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv('adult (1).csv')
df.head()


# In[3]:


df.isnull().sum()


# In[4]:


mean_capital_loss = df['capital_loss'].mean()
print(mean_capital_loss)


# In[5]:


df['capital_loss'].fillna(mean_capital_loss,inplace=True)
df.isnull().sum()


# In[6]:


df.shape


# In[7]:


df.dtypes


# In[8]:


from sklearn.preprocessing import LabelEncoder


# In[9]:


lb = LabelEncoder()


# In[10]:


df['sex'] = lb.fit_transform(df['sex'])
df['Workclass']=lb.fit_transform(df['Workclass'])
df['Workclass.1']=lb.fit_transform(df['Workclass.1'])
df['Education']=lb.fit_transform(df['Education'])
df['marital_status']=lb.fit_transform(df['marital_status'])
df['occupation']=lb.fit_transform(df['occupation'])
df['relationship']=lb.fit_transform(df['relationship'])
df['race']=lb.fit_transform(df['race'])
df['native_country']=lb.fit_transform(df['native_country'])


# In[11]:


df.dtypes


# In[12]:


x = df.iloc[:,:-1]
y = df.iloc[:,-1]
print(x.shape)
print(y.shape)


# In[13]:


from sklearn.model_selection import train_test_split


# In[14]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[15]:


from sklearn.metrics import confusion_matrix,classification_report,accuracy_score


# In[17]:


def mscore(model):
    print('Model Training Score',model.score(x_train,y_train))
    print('Model Testing Score',model.score(x_test,y_test))


def gen_metrics(ytest,ypred):
    cm = confusion_matrix(ytest,ypred)
    print(cm)
    print(classification_report(ytest,ypred))
    print(classification_error)
    print(1-accuracy_score(ytest,ypred))


# In[18]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


# In[21]:


m1 = LogisticRegression(max_iter=3000)
m2 = DecisionTreeClassifier(criterion='entropy',max_depth=14,min_samples_split=20)
m3 = RandomForestClassifier(n_estimators=50,criterion='gini',max_depth=14,min_samples_split=15)
m4 = KNeighborsClassifier(n_neighbors=15)
m5 = SVC(kernel='linear',C=1)

models = [m1,m2,m3,m4,m5]
mnames = ['LogReg','DT','RF','KNN','SVC']


# In[22]:


def mscore(model):
    print('Model Training Score',model.score(x_train,y_train))
    print('Model Testing Score',model.score(x_test,y_test))


def gen_metrics(ytest,ypred):
    cm = confusion_matrix(ytest,ypred)
    print(cm)
    print(classification_report(ytest,ypred))


# In[ ]:


d = {}
for i in range(len(models)):
    print(mnames[i])
    models[i].fit(x_train,y_train)
    mscore(models[i])
    ypred = models[i].predict(x_test)
    d[mnames[i]] = accuracy_score(y_test,ypred)
    print(ypred)
    gen_metrics(y_test,ypred)
    print('*'*70)


# In[ ]:




