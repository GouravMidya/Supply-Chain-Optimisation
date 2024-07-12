# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 08:59:28 2024

@author: goura
"""

#%% Imports

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#%% Data Loading

# partial data ./persiana order data/Orders_Vasant_2024_Q3.xlsx
df = pd.read_excel("expanded_order_data.xlsx", header=0)
df.head()

#%%
df.info()

#%% 
df.describe()

#%%
df.isnull().sum()

#%%
df['order_date'] = pd.DatetimeIndex(df['Created']).date
df['order_date'] = pd.DatetimeIndex(df['order_date'])
df['order_time'] = pd.DatetimeIndex( df['Created']).time

#%% 
df = df.drop('order_year',axis=1)

#%%
df['order_time']=df['order_time'].astype('string')
df[['Hour','Minute', 'Second']]= df['order_time'].str.split(":",expand=True)

#%%
df["Hour"].value_counts()

#%%
sns.countplot(data=df,x="Hour",palette="plasma")
plt.xticks(rotation=90)
plt.xlabel("Hour",fontsize=10,color="purple")
plt.ylabel("Frequency",fontsize=10,color="purple")
plt.title("HOUR",color="purple")
plt.show()

#%%
df['order_dayOfWeek'] = df['order_date'].dt.day_name()
df['order_dayOfWeek'].value_counts()

#%% Chart of distribution of order based on day of week
sns.countplot(data=df,x="order_dayOfWeek",palette="viridis")
plt.xticks(rotation=90)
plt.xlabel("Date",fontsize=10,color="green")
plt.ylabel("Frequency",fontsize=10,color="green")
plt.title("DATES",color="green")
plt.show()


#%% Chart of item wise quantity
fig, ax = plt.subplots(figsize=(128, 4))
sns.countplot(data=expanded_df,x="Standardized_Item",palette="tab20b_r",ax=ax)
plt.xticks(rotation=90)
plt.xlabel("Items",fontsize=10,color="black")
plt.title("Count per item",color="black")
plt.show()

#%%
with sns.axes_style('white'):
    g = sns.catplot(x="order_month", data=expanded_df, aspect=4.0, kind='count',hue='order_dates',palette="pastel")
g.set_ylabels('Frequency')
g.set_xlabels("Months")
plt.show()

#%%

# Define the order of days
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# Assuming df is your DataFrame
# Create the crosstab
ct = pd.crosstab(df["Hour"], df["order_dayOfWeek"])

# Filter out hours 1-10
ct_filtered = ct.loc[~ct.index.isin(range(1, 11))]

# Create the plot
ct_filtered.plot(kind="bar", figsize=(20, 6), 
                 color=["yellow","red","green","blue","magenta","cyan","black","orange"], 
                 title="Orders by Day and Hour",order = day_order)

plt.show()
