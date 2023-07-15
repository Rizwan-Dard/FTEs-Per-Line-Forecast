#!/usr/bin/env python
# coding: utf-8

# In[35]:


import pandas as pd

file_path = r'C:\Users\PKAhmadRi1\Desktop\Line Level Data Export No Filter Benchmarking Browser.xlsx'
df = pd.read_excel(file_path)


# In[51]:


df = df.dropna(axis=0)


# In[52]:


df.describe()


# In[53]:


df.columns


# In[54]:


df.columns = df.columns.str.replace('_', '')


# In[55]:


df.columns


# In[56]:


df.columns = df.columns.str.replace('(', '')
df.columns = df.columns.str.replace(')', '')
df.columns = df.columns.str.replace(' ', '')


# In[57]:


df.columns


# In[58]:


y = df.AverageCrewingFTE


# In[31]:


Line_features = ['Outputt', 'CapacityUtilization%',
       'AssetIntensity%', 'EstimatedUnplannedStoppages%', 'PlannedStoppages%',
       'AverageThroughputkg/h', 'ProductRelatedChangeoverLabourh', 'ZLMV%', 'FFOH-FactoryFixOverheadCHF/t',
       'FFOH-MaintenanceCHF/t', 'FFOH-OtherIndirectCostCHF/t',
       'FFOH-CommonChargesCHF/t', 'VME-VariableManufacturingCHF/t',
       'VME-VariableLabourCHF/t', 'VME-VariableEnergyCHF/t',
       'VME-OtherDirectCostCHF/t', 'VME-SubcontractingCHF/t',
       'DepreciationCHF/t', 'ConversionCostCHF/t', '3rdPartyPurchasedCHF/t',
       'RawMaterialCHF/t', 'PackingMaterialCHF/t', 'TotalMaterialCostCHF/t',
       'TotalCostofProductionCHF/t']


# In[59]:


X = df[Line_features]


# In[60]:


X.describe()


# In[63]:


from sklearn.tree import DecisionTreeRegressor


# In[64]:


DecisionTree_model = DecisionTreeRegressor(random_state=1)


# In[65]:


DecisionTree_model.fit(X, y)


# In[66]:


print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(DecisionTree_model.predict(X.head()))


# In[68]:


from sklearn.metrics import mean_absolute_error

predicted_FTEs = DecisionTree_model.predict(X)
mean_absolute_error(y, predicted_FTEs)


# In[ ]:




