#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df= pd.read_csv('Copy of Regrerssion_energy_production_data (2).csv', delimiter=';')


# In[3]:


df= df.drop_duplicates()


# In[4]:


Q1= df.quantile(0.25)
Q3= df.quantile(0.75)
IQR= Q3-Q1

df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]


# In[5]:


target= df[['energy_production']]


# In[6]:


feature= df.drop('energy_production', axis=1)


# In[7]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error


# In[8]:


x_train,x_test,y_train,y_test= train_test_split(feature, target, train_size=0.75, random_state=45)


# In[9]:


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[10]:


random_model= RandomForestRegressor()
random_model.fit(x_train,y_train)
y_pred= random_model.predict(x_test)
score= r2_score(y_test,y_pred)
mse= mean_squared_error(y_test, y_pred)
print(f'Accuracy score of the given data by using Random Forest Regressor is: {score:.2f}')
print(f'Mean Square Error of the given data by using Random Forest Regressor is: {mse:.2f}')


# In[11]:


import streamlit as st
import pickle 


# In[12]:


file= 'indraa.pkl'


# In[13]:


pickle.dump(random_model,open(file, 'wb'))


# In[14]:


model= pickle.load(open('indraa.pkl', 'rb'))


# In[15]:


st.title=('Model Deployment Using Random Forest Regressor')
st.sidebar.subheader('User Inout Parameters')


# In[16]:


def user_input_parameters():
    temperature=st.sidebar.number_input('Temperature')
    exhaust_vacuum=st.sidebar.number_input('Exhaust Input')
    amb_pressure=st.sidebar.number_input('Amb Pressure')
    r_humidity=st.sidebar.number_input('R Humidity')
    data={'temperature':temperature,'exhaust_vacuum':exhaust_vacuum,'amb_pressure':amb_pressure,'r_humidity':r_humidity}
    feature=pd.DataFrame(data,index=[0])
    return feature
df=user_input_parameters()
st.subheader('user_input_parameters')
st.write(df)

if st.button('Predict'):
    
    pred= model.predict(df)
    
    st.write(f'The Predicted Energy Production is: {pred[0]}')


# In[ ]:




