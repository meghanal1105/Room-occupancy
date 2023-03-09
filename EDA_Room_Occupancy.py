# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 22:30:57 2021

@author: 91876
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import scipy.stats
from scipy.stats import chi2_contingency
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.formula.api import ols
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn import svm
from statsmodels.stats.multicomp import pairwise_tukeyhsd


room_occupancy= pd.read_csv("C:/Users/91876/Desktop/Kaggle Individual/6_Room occupancy/occupancy.csv")
df= pd.DataFrame(room_occupancy)
df.info()
df.isna().sum()
df.describe()
df.columns
df.shape


''' Target variable '''
''' Occupancy '''
df.Occupancy.isna().sum()
df.Occupancy.value_counts().sort_index()
df.Occupancy.value_counts().sum()
df.Occupancy.describe()
df.Occupancy.unique()
''' Categorical '''
# No null values

# 0: not occupied, 1: occupied

#Countplot
sns.countplot(x ='Occupancy', data = df)




'''  1- date  ''' 
df.date.isna().sum()
df.date.value_counts().sort_index()
df.date.describe()
df.date.unique()

df.date = pd.to_datetime(df.date)
df.info()
df.date.head()
df['year'] = df["date"].dt.year
df['month'] = df["date"].dt.month_name()
df['day'] = df["date"].dt.day_name()
df['hour'] = df["date"].dt.hour
df['minute'] = df["date"].dt.minute
df.info()
 
df.year.value_counts().sort_index()
df.month.value_counts().sort_index()
df.day.value_counts().sort_index()
df.hour.value_counts().sort_index()
df.minute.value_counts().sort_index()

# Dropping the original column and 
# year column as it has only 1 year that is 2015
df = df.drop(['year','date'], axis = 1)
df.info()





'''  1(b)-  month  '''
df.month.isna().sum()
df.month.value_counts().sort_index()
df.month.value_counts().sum()
df.month.describe()
df.month.unique()
# no null values
''' Continuous '''

# Countplot
sns.countplot(x='month', data=df)


df['month'].replace('February', 2, inplace = True)
df['month'].replace('March', 3, inplace = True)
df['month'].replace('April', 4, inplace = True)
df['month'].replace('May', 5, inplace = True)
df['month'].replace('June', 6, inplace = True)
df['month'].replace('July', 7, inplace = True)
df['month'].replace('August', 8, inplace = True)
df['month'].replace('September', 9, inplace = True)
df['month'].replace('October', 10, inplace = True)
df.month.value_counts().sort_index()

# Countplot
sns.countplot(x='month', data=df)

#Histogram
sns.distplot(df.month, color = 'red')
plt.xlabel('month')
plt.title('Histogram of month')

#Boxplot
plt.boxplot(df['month'],1,'rs',1)
plt.xlabel('month')
plt.ylabel('counts')
plt.title('Boxplot of month')
# there are no outliers





'''  1(c)-  day  '''
df.day.isna().sum()
df.day.value_counts().sort_index()
df.day.describe()
df.day.unique()
# no null values
''' Continuous '''

# Countplot
sns.countplot(x='day', data=df)

df.day.value_counts().sort_index()
df['day'].replace('Monday', 1, inplace = True)
df['day'].replace('Tuesday', 2, inplace = True)
df['day'].replace('Wednesday', 3, inplace = True)
df['day'].replace('Thursday', 4, inplace = True)
df['day'].replace('Friday', 5, inplace = True)
df['day'].replace('Saturday', 6, inplace = True)
df['day'].replace('Sunday', 7, inplace = True)

# Countplot
sns.countplot(x='day', data=df)

#Histogram
sns.distplot(df.day, color = 'red')
plt.xlabel('day')
plt.title('Histogram of day')

#Boxplot
plt.boxplot(df['day'],1,'rs',1)
plt.xlabel('day')
plt.ylabel('counts')
plt.title('Boxplot of day')
# there are no outliers




'''  1(d)-  hour  '''
df.hour.isna().sum()
df.hour.value_counts().sort_index()
df.hour.describe()
df.hour.unique()
# no null values
''' Continuous '''

#Histogram
sns.distplot(df.hour, color = 'red')
plt.xlabel('hour')
plt.title('Histogram of hour')

#Boxplot
plt.boxplot(df['hour'],1,'rs',1)
plt.xlabel('hour')
plt.ylabel('counts')
plt.title('Boxplot of hour')
# there are no outliers




'''  1(e)-  minute  '''
df.minute.isna().sum()
df.minute.value_counts().sort_index()
df.minute.describe()
df.minute.unique()
# no null values
''' Continuous '''

#Histogram
sns.distplot(df.minute, color = 'red')
plt.xlabel('minute')
plt.title('Histogram of minute')

#Boxplot
plt.boxplot(df['minute'],1,'rs',1)
plt.xlabel('minute')
plt.ylabel('counts')
plt.title('Boxplot of minute')
# there are no outliers





'''  2-  Temperature  '''
df.Temperature.isna().sum()
df.Temperature.value_counts().sort_index()
df.Temperature.describe()
df.Temperature.unique()
# no null values
''' Continuous '''

#Histogram
sns.distplot(df.Temperature, color = 'red')
plt.xlabel('Temperature')
plt.title('Histogram of Temperature')

#Boxplot
plt.boxplot(df['Temperature'],1,'rs',1)
plt.xlabel('Temperature')
plt.ylabel('counts')
plt.title('Boxplot of Temperature')
# there are outliers

# Outliers Count
IQR2 = df['Temperature'].quantile(0.75) - df['Temperature'].quantile(0.25)
IQR2

UL2 = df['Temperature'].quantile(0.75) + (1.5*IQR2)
UL2

df.Temperature[(df.Temperature > UL2)].value_counts().sum()
# 35

df.Temperature = np.where(df.Temperature > UL2, UL2, df.Temperature)

df.Temperature[(df.Temperature > UL2)].value_counts().sum()
# 0





'''  3-  Humidity  '''
df.Humidity.isna().sum()
df.Humidity.value_counts().sort_index()
df.Humidity.describe()
df.Humidity.unique()
# no null values
''' Continuous '''

#Histogram
sns.distplot(df.Humidity, color = 'red')
plt.xlabel('Humidity')
plt.title('Histogram of Humidity')

#Boxplot
plt.boxplot(df['Humidity'],1,'rs',1)
plt.xlabel('Humidity')
plt.ylabel('counts')
plt.title('Boxplot of Humidity')
# there are outliers

# Outliers Count
IQR3 = df['Humidity'].quantile(0.75) - df['Humidity'].quantile(0.25)
IQR3

UL3 = df['Humidity'].quantile(0.75) + (1.5*IQR3)
UL3

df.Humidity[(df.Humidity > UL3)].value_counts().sum()
# 71

df.Humidity = np.where(df.Humidity > UL3, UL3, df.Humidity)

df.Humidity[(df.Humidity > UL3)].value_counts().sum()
# 0



'''  4-  Light  '''
df.Light.isna().sum()
df.Light.value_counts().sort_index()
df.Light.describe()
df.Light.unique()
# no null values
''' Continuous '''

#Histogram
sns.distplot(df.Light, color = 'red')
plt.xlabel('Light')
plt.title('Histogram of Light')

#Boxplot
plt.boxplot(df['Light'],1,'rs',1)
plt.xlabel('Light')
plt.ylabel('counts')
plt.title('Boxplot of Light')
# there are outliers

# Outliers Count
IQR4 = df['Light'].quantile(0.75) - df['Light'].quantile(0.25)
IQR4

UL4 = df['Light'].quantile(0.75) + (1.5*IQR4)
UL4

df.Light[(df.Light > UL4)].value_counts().sum()
# 5

df.Light = np.where(df.Light > UL4, UL4, df.Light)

df.Light[(df.Light > UL4)].value_counts().sum()
# 0





'''  5-  CO2  '''
df.CO2.isna().sum()
df.CO2.value_counts().sort_index()
df.CO2.describe()
df.CO2.unique()
# no null values
''' Continuous '''

#Histogram
sns.distplot(df.CO2, color = 'red')
plt.xlabel('CO2')
plt.title('Histogram of CO2')

#Boxplot
plt.boxplot(df['CO2'],1,'rs',1)
plt.xlabel('CO2')
plt.ylabel('counts')
plt.title('Boxplot of CO2')
# there are outliers

# Outliers Count
IQR5 = df['CO2'].quantile(0.75) - df['CO2'].quantile(0.25)
IQR5

UL5 = df['CO2'].quantile(0.75) + (1.5*IQR5)
UL5

df.CO2[(df.CO2 > UL5)].value_counts().sum()
# 618

df.CO2 = np.where(df.CO2 > UL5, UL5, df.CO2)

df.CO2[(df.CO2 > UL5)].value_counts().sum()
# 0




'''  6- HumidityRatio  '''
df.HumidityRatio.isna().sum()
df.HumidityRatio.value_counts().sort_index()
df.HumidityRatio.describe()
df.HumidityRatio.unique()
# no null values
''' Continuous '''

#Histogram
sns.distplot(df.HumidityRatio, color = 'red')
plt.xlabel('HumidityRatio')
plt.title('Histogram of HumidityRatio')

#Boxplot
plt.boxplot(df['HumidityRatio'],1,'rs',1)
plt.xlabel('HumidityRatio')
plt.ylabel('counts')
plt.title('Boxplot of HumidityRatio')
# there are outliers

# Outliers Count
IQR6 = df['HumidityRatio'].quantile(0.75) - df['HumidityRatio'].quantile(0.25)
IQR6

UL6 = df['HumidityRatio'].quantile(0.75) + (1.5*IQR6)
UL6

df.HumidityRatio[(df.HumidityRatio > UL6)].value_counts().sum()
# 208

df.HumidityRatio = np.where(df.HumidityRatio > UL6, UL6, df.HumidityRatio)

df.HumidityRatio[(df.HumidityRatio > UL6)].value_counts().sum()
# 0

''' EDA is done '''


df.to_csv('C:/Users/91876/Desktop/Kaggle Individual/6_Room occupancy/Exported files/EDA_Room_Occupancy.csv')
