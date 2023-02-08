#!/usr/bin/env python
# coding: utf-8

# In[59]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn import tree


# In[60]:


df = pd.read_csv("titanic.csv")


# In[61]:


df.head(50)


# In[62]:


inputs = df.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'],axis = 'columns')


# In[63]:


inputs


# In[64]:


target = inputs.Survived


# In[65]:


target


# In[66]:


inputs


# In[67]:


sex_l = LabelEncoder() 


# In[68]:


inputs['Sex_n'] = sex_n.fit_transform(inputs['Sex'])


# In[69]:


inputs


# In[70]:


inputs = inputs.drop(['Sex','Survived'],axis = "columns" )


# In[71]:


inputs


# In[72]:


inputs.Age = inputs.Age.fillna(inputs.Age.mean())


# In[73]:


model = tree.DecisionTreeClassifier()


# In[74]:


model.fit(inputs,target)


# In[76]:


model.score(inputs,target)*100


# In[ ]:




