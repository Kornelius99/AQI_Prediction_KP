#!/usr/bin/env python
# coding: utf-8

# ### Air Quality Index Prediction

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import warnings
warnings.filterwarnings('ignore')
import pickle


# In[5]:


data = pd.read_csv("AQI_Data.csv",skipinitialspace = True)
data = pd.DataFrame(data)


# In[6]:


data = data.sort_values('date')


# In[7]:


data.head()


# In[8]:


data.shape


# Chemical compounds that causes for Air pollution
# 
# 1) PM2.5 (Particulate Matter 2.5-micrometer)                                    
# 2) PM10 (Particulate Matter 10-micrometer)                                           
# 3) O3 (Ozone or Trioxygen)                                              
# 4) NO2 (Any Nitric x-oxide)                                           
# 5) SO2 (Sulphur Dioxide)                                                     
# 6) CO (Carbon Monoxide)                       
# 
# Air Quality Level and its Level of Health Concern
# 
# 1) 0 -50 ------- Good                                                        
# 2) 51 - 100 ----- Satisfactory                                                              
# 3) 101 - 200 ---- Moderate - Unhealthy for sensitive people                                         
# 4) 201 - 300 ---- Poor - Unhealthy                              
# 5) 301 - 400 ---- Very Poor - Very Unhealthy                                       
# 6) 400+ --------- Severe - Hazardous

#  

# In[9]:


data.isnull().sum()  # checking the null values


# In[10]:


sns.heatmap(data.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# As per the above graph, we can see that there are no null values found the in the data set.

#  

# In[11]:


# date split
data["year"]=data['date'].apply(lambda x:x.split("-")[2])
data["Month"]=data['date'].apply(lambda x:x.split("-")[1])
data["date"]=data['date'].apply(lambda x:x.split("-")[0])


# In[12]:


data.head()


# In[13]:


data.info()


# In[14]:


data.describe()


#  

# In[15]:


plt.figure(figsize=(19,5))
aqilevel = sns.catplot(x="year",y='AQI', data=data, hue='AQI_Level',height=6, aspect=10/5)
plt.ylim(0,1200)
plt.title('yearly AQI level')
aqilevel.set_xticklabels(rotation=90)


# From the above graph,                                     
# AQI value is decreasing in recent years as we can see the Good & Satisfactory values occurring from the year 2020. 
# Sample values from above 400(indicated in green color) are decreasing by year which indicates better air quality.

#  

# In[16]:


from sklearn import preprocessing


# In[17]:


lblencode = preprocessing.LabelEncoder()
data['AQI_Level_numeric'] = lblencode.fit_transform(data['AQI_Level'])


# In[18]:


data['AQI_Level'].unique()


# In[19]:


data.head()


# In[20]:


data.columns


# In[21]:


data_city_day = data.copy()
data_city_day.columns


# In[22]:


data.corr()


# In[23]:


plt.figure(figsize=(10,6))
sns.heatmap(data.corr(),cmap='RdYlGn',annot=True);


# Box plots are showing correlation between the molecules from the above correlation heatmap.
# 

#  

# In[24]:


citywise_AQI = data[['City','AQI']].groupby(['City']).mean().sort_values(['AQI']).reset_index()
citywise_AQI.head()
plt.figure(figsize=(14,3))
sns.set(font_scale=1.5)
sns.barplot(x='City', y='AQI', data=citywise_AQI).set(title ='City wise AQI')
plt.xticks(rotation=90)
plt.show()


# Delhi is the highest polluted city followed by patna and Gurgram as per the above chart.

#  

# In[25]:


x=data['year']
y=data['AQI']


# In[26]:


#Yearly AQI Trend
fig, ax = plt.subplots(figsize=(14, 5))
plt.title('Yearly AQI')
barchart=sns.barplot(x = x, y =y, ax=ax)
barchart.bar_label(ax.containers[0],label_type='edge',fmt='%.0f',padding=5)
plt.show()


# From the above barchart, we cleary say that the aqi value is higher in the year 2014 foloowed by 2015 etc,and      
# Trend indicating that a better aqi from 2020 due to envirornmental changes

#  

# In[24]:


pollutants = data.iloc[:,2:8]


# In[25]:


#Yearly Trend of Pollutants.
for pol in pollutants:
    plt.figure(figsize=(14,4))
    sns.lineplot(x='year', y=pol,data=data.sort_values("year"))
    plt.title(f"Yearly {pol} - Pollutant Trend ", fontsize=14)
    plt.show()


# # Model Building

# In[26]:


X  =  data.loc[:,"pm25":"co"]
X.head()


# In[27]:


y  =  data.iloc[:,8:9].values


# In[28]:


y


# ## Train Test Split

# In[29]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# ### Linear Regression

# In[30]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score


# In[31]:


m1lr = LinearRegression()
m1lr.fit(X_train, y_train)


# In[32]:


round(m1lr.score(X_train,y_train)*100,2)


# In[33]:


lnrscore = round(m1lr.score(X_test,y_test)*100,2)
print(lnrscore)


# In[34]:


m1lr.predict(X_test)


# #### Cross-Validation

# In[35]:


from sklearn.model_selection import cross_val_score
lr_cvscore = cross_val_score(m1lr,X_train,y_train,cv=5)
round(lr_cvscore.mean()*100,2)


# #### Model Evaluation

# In[36]:


lnr_prediction = m1lr.predict(X_test)


# In[37]:


sns.distplot(y_test-lnr_prediction,kde=True)


# In[38]:


from sklearn import metrics


# In[39]:


print('MAE:', metrics.mean_absolute_error(y_test, lnr_prediction))
print('MSE:', metrics.mean_squared_error(y_test, lnr_prediction))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, lnr_prediction)))


# ### Decision Tree Regressor

# In[40]:


from sklearn.tree import DecisionTreeRegressor


# In[41]:


dtr = DecisionTreeRegressor(criterion='mse')


# In[42]:


dtr.fit(X_train,y_train)


# In[43]:


round(dtr.score(X_train,y_train)*100,2)


# In[44]:


dtrscore = round(dtr.score(X_test,y_test)*100,2)
dtrscore


# #### Cross Validation

# In[45]:


dt_cvscore = cross_val_score(dtr,X_train,y_train,cv=5)
round(dt_cvscore.mean()*100,2)


# #### Model Evaluation

# In[46]:


dt_prediction=dtr.predict(X_test)
dt_prediction


# In[47]:


print('MAE:', metrics.mean_absolute_error(y_test, dt_prediction))
print('MSE:', metrics.mean_squared_error(y_test, dt_prediction))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, dt_prediction)))


# ## Xgboost Regressor

# In[48]:


#pip install xgboost


# In[49]:


import xgboost as xgb


# In[50]:


xgbr = xgb.XGBRegressor()
xgbr.fit(X_train,y_train)


# In[51]:


round(xgbr.score(X_train, y_train)*100,2)


# In[52]:


xgbscore =round(xgbr.score(X_test, y_test)*100,2)
xgbscore


# In[53]:


round(xgbr.score(X, y)*100,2)


# In[54]:


#Cross Validation
xb_cvscore=cross_val_score(xgbr,X_train,y_train,cv=5)
round(xb_cvscore.mean()*100,2)


# In[55]:


xgbr_prediction=xgbr.predict(X_test)


# In[56]:


print('MAE:', metrics.mean_absolute_error(y_test, xgbr_prediction))
print('MSE:', metrics.mean_squared_error(y_test, xgbr_prediction))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, xgbr_prediction)))


# ### KNN Regressor

# In[57]:


from sklearn.neighbors import KNeighborsRegressor


# In[58]:


knr_regressor=KNeighborsRegressor(n_neighbors=2)
knr_regressor.fit(X_train,y_train)


# In[59]:


round(knr_regressor.score(X_train,y_train)*100,2)


# In[60]:


knnscore=round(knr_regressor.score(X_test,y_test)*100,2)
knnscore


# In[61]:


round(knr_regressor.score(X,y)*100,2)


# In[62]:


#cross validation
score = cross_val_score(knr_regressor,X_train,y_train,cv=5)
round(score.mean()*100,2)


# In[63]:


#Model Evaluation

knr_prediction=knr_regressor.predict(X_test)
print('MAE:', metrics.mean_absolute_error(y_test, knr_prediction))
print('MSE:', metrics.mean_squared_error(y_test, knr_prediction))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, knr_prediction)))


# ### Support Vector Machine

# In[64]:


from sklearn.svm import SVR


# In[65]:


svmr = SVR()
svmr.fit(X_train,y_train)


# In[66]:


round(svmr.score(X_train,y_train)*100,2)


# In[67]:


svmscore=round(svmr.score(X_test,y_test)*100,2)
svmscore


# In[68]:


round(svmr.score(X,y)*100,2)


# In[69]:


#cross validation
svm_score = cross_val_score(svmr,X_train,y_train,cv=5)
round(svm_score.mean()*100,2)


# In[70]:


#Model Evaluation

svm_prediction=svmr.predict(X_test)
print('MAE:', metrics.mean_absolute_error(y_test, svm_prediction))
print('MSE:', metrics.mean_squared_error(y_test, svm_prediction))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, svm_prediction)))


# ### Best Accuracy Model

# In[71]:


import pandas as pd
accuracy_data = pd.DataFrame.from_dict({'Model name': ['Linear Regression','Decision Tree','XGBoost','KNN','SVM'],
       'Accuracy': [lnrscore,dtrscore,xgbscore,knnscore,svmscore]})


# In[72]:


print(accuracy_data)


# In[73]:


import seaborn as sns
Model_name=['Linear Regression','Decision Tree','XGBoost','KNN','SVM']
x=Model_name
y=[lnrscore,dtrscore,xgbscore,knnscore,svmscore]
df= pd.DataFrame({"Model_name":Model_name,"y":y})
fig, ax = plt.subplots(figsize=(10, 5))
barchart = sns.barplot(x=x, y=y, ax=ax,data=df,order=df.sort_values('y',ascending=False).Model_name )
plt.xlabel('Model Name')
plt.ylabel('Accuracy value')
plt.title('Model wise Accuracy Scores')
barchart.bar_label(ax.containers[0], label_type='edge')


# ##### XGBoost Algorithm is the best accuracy model when comparing with other regression models

# ### AQI Forecasting Using Prophet

# In[27]:


data_f = pd.read_csv('AQI_Data.csv',skipinitialspace = True)
data_f = pd.DataFrame(data_f)
data_f = data_f.sort_values('date')
data_f.head()


# In[28]:


data_f['date'] = pd.to_datetime(data_f['date'], errors = 'coerce')


# In[29]:


data_f = data_f.groupby('date')['AQI'].sum().reset_index()


# In[30]:


data_f.head()


# In[31]:


data_f = data_f.set_index('date')
data_f.index


# In[32]:


data_f.head()


# In[33]:


y = data_f['AQI'].resample('MS').mean()


# In[34]:


y


# In[35]:


y.plot(figsize=(15, 6))
plt.title('Yearly AQI Trend')
plt.xlabel('Year')
plt.ylabel('AQI-Value')
plt.show()


# According to the above graph, the annual aqi mean trend decreased from 2014 to 2017 and increased from 2018. In 2021, we can observe that the greater aqi mean.

#  

# In[36]:


data_f = data_f.reset_index()


# In[37]:


from prophet import Prophet


# In[38]:


data_f = data_f.rename(columns={'date': 'ds', 'AQI': 'y'})
data_f_model = Prophet(interval_width=0.95)


# In[39]:


data_f


# In[40]:


data_f_model.fit(data_f)


# In[41]:


AQI_forecast = data_f_model.make_future_dataframe(periods=48, freq='MS')
AQI_forecast = data_f_model.predict(AQI_forecast)


# In[42]:


plt.figure(figsize=(18, 6))
data_f_model.plot(AQI_forecast, xlabel = 'Date', ylabel = 'AQI')
plt.title('AQI Forecast');


# From the above graph we can see the yearly aqi forecasting trend line up to 2026.

#  

# In[90]:


# save the model to disk
filename = 'AQI_forecasting_model.sav'
pickle.dump(data_f_model, open(filename, 'wb'))


# In[91]:


# save the model to disk
filename = 'AQI_Prediction_model.sav'
pickle.dump(xgbr, open(filename, 'wb'))

