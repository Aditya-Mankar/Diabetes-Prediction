#!/usr/bin/env python
# coding: utf-8

# # Description:                                                                                                                   
# The objective of the dataset is to diagnostically predict whether or not a patient has diabetes, based on certain diagnostic measurements included in the dataset.
# 
# Attributes:
# 1. Glucose Level
# 2. BMI
# 3. Blood pressure
# 4. Pregnancies
# 5. Skin thickness
# 6. Insulin
# 7. Diabetes pedigree function
# 8. Age
# 9. Outcome

# # Step 0: Import libraries and Dataset

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

import pickle

# In[2]:


dataset = pd.read_csv('diabetes.csv')



# # Step 3: Data Preprocessing

# In[13]:


dataset_X = dataset.iloc[:,[1, 4, 5, 7]].values
dataset_Y = dataset.iloc[:,8].values


# In[14]:


dataset_X


# In[15]:


from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
dataset_scaled = sc.fit_transform(dataset_X)


# In[16]:


dataset_scaled = pd.DataFrame(dataset_scaled)


# In[17]:


X = dataset_scaled
Y = dataset_Y


# In[18]:


X


# In[19]:


Y


# In[20]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 42, stratify = dataset['Outcome'] )


# # Step 4: Data Modelling

# In[25]:


from sklearn.svm import SVC
svc = SVC(kernel = 'linear', random_state = 42)
svc.fit(X_train, Y_train)


# In[26]:


svc.score(X_test, Y_test)


# In[27]:


Y_pred = svc.predict(X_test)





pickle.dump(svc, open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))
#print(model.predict(sc.transform(np.array([[86, 66, 26.6, 31]]))))


