#!/usr/bin/env python
# coding: utf-8

# # INTRAINZ EDUTECH - Data Science Internship Project
# 
# # Recommender System
# 

# # Import Libraries

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


# Import Dataset
data = pd.read_csv('C:/Users/Data Analyst/OneDrive/Documents/online retail.csv')


# In[3]:


data


# In[4]:


# Shape of our Dataset
print('Shape of our Dataset is',data)


# In[5]:


data.info()


# In[6]:


# Eplore the unique values of each attribute and print it.
print('Number of transcations: ',data['InvoiceNo'].nunique())
print('Number of Products: ',data['StockCode'].nunique())
print('Number of Customers: ',data['CustomerID'].nunique())
print('Number of Countries: ',data['Country'].nunique())


# # Determine the percentage of customers with null id

# In[7]:


# Formula to compute percentage of customers in this dataset
print('Percentage of customers NA: ',round(data['CustomerID'].isnull().sum() * 100 / len(data),2),'%')


# In[8]:


# Analysis on Quantative Data
data.describe()


# # Determine the percentage of Cancelled Orders

# In[9]:


# Get cancelled transactions
cancelled_orders = data[data['InvoiceNo'].astype(str).str.contains('C')]
cancelled_orders.head()
# search for transaction where quantity == 80995
cancelled_orders[cancelled_orders['Quantity']==-80995]


# In[10]:


# Group the Customers by country


# In[11]:


# check how many rows our dataframes contains cancelled orders
print('We have ',len(cancelled_orders), 'cancelled orders.')


# In[12]:


# percentage of cancelled orders in total orders
total_orders = data['InvoiceNo'].nunique()
cancelled_number = len(cancelled_orders)
percentage_cancelled = (cancelled_number / total_orders) * 100
print('% of orders cancelled: {}/{} ({:.2f}%)'.format(cancelled_number,total_orders,percentage_cancelled))


# # Analyse country with more customers

# In[13]:


# Customers by country
data['total_cost'] = data['Quantity'] = data['UnitPrice']
data.head()


# In[14]:


data.groupby('Country').sum().sort_values(by='UnitPrice',ascending=False)


# # Find Popoular items using Graphs and Pivot table

# # Popular items by Globally

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from matplotlib.colors import ListedColormap


# In[2]:


data = pd.read_csv('C:/Users/Data Analyst/OneDrive/Documents/online retail.csv')
df=data


# In[17]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Create a pivot table to get counts of quantities for each product globally
global_popular_items = pd.pivot_table(df, values='Quantity', index='Description', aggfunc='sum').sort_values(by='Quantity', ascending=False)

# Plot globally popular items
plt.figure(figsize=(10, 5))
sns.barplot(x=global_popular_items.index, y=global_popular_items['Quantity'], palette='viridis')
plt.title('Globally Popular Items')
plt.xlabel('Product Description')
plt.ylabel('Total Quantity Sold')
plt.show()


# # Popular items by Country_wise

# In[19]:


# Create a pivot table to get counts of quantities for each product country-wise
country_wise_popular_items = pd.pivot_table(df, values='Quantity', index=['Country', 'Description'], aggfunc='sum').sort_values(by=['Country', 'Quantity'], ascending=[True, False]).reset_index()

# Plot country-wise popular items
plt.figure(figsize=(15, 5))
sns.barplot(x='Description', y='Quantity', hue='Country', data=country_wise_popular_items, palette='viridis')
plt.title('Country-wise Popular Items')
plt.xlabel('Product Description')
plt.ylabel('Total Quantity Sold')
plt.show()


# # Popular items by Month_wise

# In[ ]:


# Convert 'InvoiceDate' column to datetime format
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Create a pivot table to get counts of quantities for each product month-wise
month_wise_popular_items = pd.pivot_table(df, values='Quantity', index=['InvoiceDate', 'Description'], aggfunc='sum').sort_values(by=['InvoiceDate', 'Quantity'], ascending=[True, False]).reset_index()

# Plot month-wise popular items
plt.figure(figsize=(15, 5))
sns.barplot(x='Description', y='Quantity', hue=month_wise_popular_items['InvoiceDate'].dt.month, data=month_wise_popular_items, palette='viridis')
plt.title('Month-wise Popular Items')
plt.xlabel('Product Description')
plt.ylabel('Total Quantity Sold')
plt.show()


# # Functions to analyze and Print Recommendations

# In[ ]:


import pandas as pd

recommendations_data = pd.read_csv('C:/Users/Data Analyst/OneDrive/Documents/online retail.csv')
recommendations_df = pd.DataFrame(recommendations_data)


# In[ ]:


def get_customer_recommendations(customer_id, top_n=5):
    customer_recommendations = recommendations_df[recommendations_df['CustomerID'] == customer_id].nlargest(top_n, 'UnitPrice')
    return customer_recommendations

def print_recommendations(customer_id, top_n=5):
    customer_recommendations = get_customer_recommendations(customer_id, top_n)
    
    if customer_recommendations.empty:
        print(f"No recommendations found for Customer {customer_id}.")
    else:
        print(f"Top {top_n} recommendations for Customer {customer_id}:")
        print(customer_recommendations[['StockCode', 'UnitPrice']])


# In[ ]:


# real Dataset Recommendations
customer_id_to_analyze =1
top_n_recommendations = 541910

print_recommendations(customer_id_to_analyze, top_n_recommendations)


# In[ ]:





# In[ ]:




