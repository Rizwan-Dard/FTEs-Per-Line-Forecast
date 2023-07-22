#!/usr/bin/env python
# coding: utf-8

# In[8]:


from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


# In[25]:


import pandas as pd

df = pd.read_excel(r'C:\Users\PKAhmadRi1\Desktop\AOA_Resource_Test.xlsx')
df = df.dropna(axis=0)


# In[26]:


df.columns


# In[27]:


df.columns = df.columns.str.replace('_', '')
df.columns = df.columns.str.replace(')', '')
df.columns = df.columns.str.replace('(', '')
df.columns = df.columns.str.replace(' ', '')

df.columns


# In[28]:


features = ['SKUs#', 'Outputt', 'CapacityUtilization%',
       'AssetIntensity%', 'EstimatedUnplannedStoppages%', 'PlannedStoppages%',
       'AverageThroughputkg/h', 'ProductRelatedChangeoverLabourh',
       'AverageCrewingFTE', 'ZLMV%']

X = df[features]
y = df.OKNotOK


# In[35]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[36]:


from sklearn.svm import SVC
svm_model_linear = SVC(kernel = 'linear', C = 1).fit(X_train, y_train)
svm_predictions = svm_model_linear.predict(X_test)
  
# model accuracy for X_test  
accuracy = svm_model_linear.score(X_test, y_test)
  
# creating a confusion matrix
cm = confusion_matrix(y_test, svm_predictions)


# In[37]:


cm


# In[39]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cm), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# In[ ]:




