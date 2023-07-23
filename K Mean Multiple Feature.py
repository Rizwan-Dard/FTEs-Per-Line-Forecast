#!/usr/bin/env python
# coding: utf-8

# In[4]:


from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import pandas as pd

df = pd.read_excel(r'C:\Users\PKAhmadRi1\Desktop\AOA_Resource_Test.xlsx')
df = df.dropna(axis=0)


# In[8]:


df.columns = df.columns.str.replace('_', '')
df.columns = df.columns.str.replace(')', '')
df.columns = df.columns.str.replace('(', '')
df.columns = df.columns.str.replace(' ', '')
df.columns = df.columns.str.replace('/', '')
df.columns = df.columns.str.replace('#', '')
df.columns = df.columns.str.replace('%', '')                                    
df.columns


# In[18]:


#y=df.AverageCrewingFTE


# In[28]:


#X=df.ConversionCostCHFt


# In[30]:


#plt.scatter(X, y)
#plt.show()


# In[14]:


from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3)

y = kmeans.fit_predict(df[['SKUs', 'Outputt', 'CapacityUtilization',
       'AssetIntensity']])

df['Cluster'] = y

print(df)

#data = list(zip(X, y))
#inertias = []

#for i in range(1,11):
 #   kmeans = KMeans(n_clusters=i)
  #  kmeans.fit(data)
   # inertias.append(kmeans.inertia_)

# plt.plot(range(1,11), inertias, marker='o')
# plt.title('Elbow method')
# plt.xlabel('Number of clusters')
# plt.ylabel('Inertia')
# plt.show()


# In[1]:


# kmeans = KMeans(n_clusters=3)
# kmeans.fit(data)

# plt.scatter(X, y, c=kmeans.labels_)
# plt.show()


# In[ ]:





# In[ ]:




