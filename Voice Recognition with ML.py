#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


df = pd.read_excel('voice.xlsx')
df.head()


# In[4]:


df.shape


# In[5]:


df.isnull().sum()


# In[6]:


data_pie=df['label'].value_counts().rename_axis('label').reset_index(name='gender_count')


# In[8]:


plt.figure(figsize=(10,10))
plt.pie(data_pie.gender_count,labels=data_pie.label,startangle=90,autopct='%.1f%%')
plt.title('Gender of voices')
plt.show()


# In[9]:


df.dtypes


# In[10]:


from sklearn.preprocessing import LabelEncoder


# In[11]:


lb = LabelEncoder()


# In[12]:


df['label'] = lb.fit_transform(df['label'])


# In[13]:


df.dtypes


# In[14]:


x = df.iloc[:,:-1]
y = df.iloc[:,-1]
print(x.shape)
print(y.shape)


# In[15]:


from sklearn.model_selection import train_test_split


# In[16]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[17]:


from sklearn.metrics import confusion_matrix,classification_report,accuracy_score


# In[18]:


def mscore(model):
    print('Model Training Score',model.score(x_train,y_train))
    print('Model Testing Score',model.score(x_test,y_test))


def gen_metrics(ytest,ypred):
    cm = confusion_matrix(ytest,ypred)
    print(cm)
    print(classification_report(ytest,ypred))


# In[19]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


# In[20]:


m1 = LogisticRegression(max_iter=3000)
m2 = DecisionTreeClassifier(criterion='entropy',max_depth=14,min_samples_split=20)
m3 = RandomForestClassifier(n_estimators=50,criterion='gini',max_depth=14,min_samples_split=15)
m4 = KNeighborsClassifier(n_neighbors=15)
m5 = SVC(kernel='linear',C=1)

models = [m1,m2,m3,m4,m5]
mnames = ['LogReg','DT','RF','KNN','SVC']


# In[21]:


def mscore(model):
    print('Model Training Score',model.score(x_train,y_train))
    print('Model Testing Score',model.score(x_test,y_test))


def gen_metrics(ytest,ypred):
    cm = confusion_matrix(ytest,ypred)
    print(cm)
    print(classification_report(ytest,ypred))


# In[22]:


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


# In[27]:


#LOGREG is the model with best accuracy with minimum difference of training score and testing score


# In[ ]:




