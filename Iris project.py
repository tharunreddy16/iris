#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.read_csv('Iris1.csv')
print(df)


# In[3]:


df.describe()


# In[7]:


data = df.values
X = data[:, 1:4]
Y = data[:, 5]


# In[8]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)


# In[9]:


from sklearn.svm import SVC
svm = SVC()
svm.fit(X_train, y_train)


# In[10]:


predictions = svm.predict(X_test)


# In[11]:


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, predictions) * 100
print('Accuracy: %.2f' % accuracy)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




