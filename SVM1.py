#!/usr/bin/env python
# coding: utf-8

# In[8]:


from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


# In[9]:


import pandas as pd

df = pd.read_csv(r'C:\Users\PKAhmadRi1\Desktop\iris.data.csv')


# In[10]:


df.columns


# In[16]:


features = ['A', 'B', 'C', 'D']


# In[17]:


X=df[features]
y = df.FT


# In[18]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)


# In[19]:


from sklearn.svm import SVC
svm_model_linear = SVC(kernel = 'linear', C = 1).fit(X_train, y_train)
svm_predictions = svm_model_linear.predict(X_test)
  
# model accuracy for X_test  
accuracy = svm_model_linear.score(X_test, y_test)
  
# creating a confusion matrix
cm = confusion_matrix(y_test, svm_predictions)


# In[20]:


cm


# In[ ]:




