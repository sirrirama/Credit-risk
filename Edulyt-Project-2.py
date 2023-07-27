#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


data = pd.read_csv(r"C:\Users\sriram kumar\Dropbox\PC\Downloads\Credit Banking-2 edulyt.csv")


# In[4]:


data.head()


# In[6]:


data.tail()


# In[7]:


print(f"The given dataset contains {data.shape[0]} rows and {data.shape[1]} columns")


# In[8]:


print(f'-----\n{data.dtypes.value_counts()}')


# In[10]:


df=pd.read_csv(r"C:\Users\sriram kumar\Dropbox\PC\Desktop\Customer_info.csv")


# In[11]:


df.head()


# In[12]:


print(f"The given dataset contains {df.shape[0]} rows and {df.shape[1]} columns")


# In[13]:


print(f'-----\n{df.dtypes.value_counts()}')


# ## 1.Calculate the spend in terms of Product, State and Payment method .

# In[15]:


transactions = pd.DataFrame(data)


# In[23]:


Spend_on_product= transactions.groupby(['P_CATEGORY'])['Selling_price'].sum().reset_index()

Spend_on_product


# In[24]:


sns.barplot(x="P_CATEGORY", y="Selling_price", data=transactions)
plt.xlabel("Product Category")
plt.ylabel("Total Selling Price")
plt.title("Total Selling Price per Product Category")
plt.xticks(rotation=45)  # Rotates x-axis labels for better readability
plt.show()


# In[25]:


Spend_by_paymentmethod=transactions.groupby(['Payment method'])['Selling_price'].sum().reset_index()

Spend_by_paymentmethod


# In[26]:


sns.barplot(x="Payment method", y="Selling_price", data=transactions)
plt.xlabel("Payment method")
plt.ylabel("Total Selling Price")
plt.title("Total Selling Price per Payment method")
plt.xticks(rotation=45)  # Rotates x-axis labels for better readability
plt.show()


# In[27]:


transactions1 = pd.DataFrame(df)


# In[29]:


Spend_State_wise=transactions1.groupby(['State'])['Selling Price'].sum().reset_index()

Spend_State_wise


# In[33]:


sns.swarmplot(x="State", y="Selling Price", data=Spend_State_wise)
plt.xlabel("State")
plt.ylabel("Total Selling Price")
plt.title("Total Selling Price per Product Category (Swarm Plot)")
plt.xticks(rotation=45)
plt.show()


# ## 2.Calculate the highest 5 spending in all above categories.

# In[36]:


sorted_data1=Spend_on_product.sort_values('Selling_price', ascending=False)

sorted_data1.head()


# In[35]:


top_entries = sorted_data1.head(5)  

plt.bar(top_entries['P_CATEGORY'], top_entries['Selling_price'], color='skyblue')

plt.xlabel("Product Category")
plt.ylabel("Total Selling Price")
plt.title("Top 5 Product Categories by Total Selling Price (Bar Plot)")
plt.xticks(rotation=45)

plt.show()


# In[38]:


sorted_data2=Spend_by_paymentmethod.sort_values('Selling_price', ascending=False)

sorted_data2.head()


# In[47]:


import plotly.graph_objects as go

top_entries = sorted_data2.head(5)   
fig = go.Figure()

fig.add_trace(go.Bar(
    x=top_entries['Payment method'],
    y=top_entries['Selling_price'],
    marker_color='yellowgreen',
))


fig.update_layout(
    xaxis_title="Payment method",
    yaxis_title="Total Selling Price",
    title="Top 5 Product Categories by Total Selling Price (Bar Plot)",
    xaxis_tickangle=-45,  # Rotate x-axis labels for better readability
)

fig.show()


# In[49]:


sorted_data3=Spend_State_wise.sort_values('Selling Price', ascending=False)

sorted_data3.head()


# In[53]:


from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource
from bokeh.transform import factor_cmap
from bokeh.palettes import Spectral11
from bokeh.io import output_notebook

top_entries = sorted_data3.head(5)  

source = ColumnDataSource(top_entries)

p = figure(x_range=top_entries['State'], plot_height=400, plot_width=800,
           title="Top 5 States by Total Selling Price (Bar Plot)", toolbar_location=None, tools="")

p.vbar(x='State', top='Selling Price', width=0.6, source=source, line_color="white",
       fill_color=factor_cmap('State', palette=Spectral11, factors=top_entries['State']))

p.xaxis.major_label_orientation = 45
p.xaxis.axis_label = "State"
p.yaxis.axis_label = "Total Selling Price"

p.xgrid.grid_line_color = None
p.ygrid.grid_line_color = None

output_notebook()  
show(p)


# ## 3. Give your opinion on return category like customers returning the products belongs to which state,age group, condition, category of the product or is it related to discount.

# In[54]:


fd = data.groupby("Credit_card").sum()['Return_ind'].reset_index()


# In[55]:


df["State"]


# In[56]:


fd['State']= df['State']


# In[57]:


fd


# In[60]:


sorted_data =fd.sort_values('Return_ind', ascending=False)
sorted_data


# In[65]:



top_entries = sorted_data.head(5)  
ax = top_entries.plot(kind='bar', x='State', y='Return_ind', color='skyblue',
                      legend=False, figsize=(10, 6))

ax.set_xlabel("X Axis Label")
ax.set_ylabel("Return_ind")
ax.set_title("Top 5 Entries by Return_ind (Bar Plot)")

ax.set_xticklabels(top_entries['State'], rotation=45)

plt.show()


# ####  Here we can see that state with highest return index are Massachusetts,Arizona,Texas and California	

# In[66]:


df["Customer Segment"]


# In[67]:


fd['Customer Segment']=df['Customer Segment']


# In[68]:


fd


# In[69]:


sorted_data1 =fd.sort_values('Return_ind', ascending=False)
sorted_data1


# In[74]:


top_entries = sorted_data1.head(5)  
ax = top_entries.plot(kind='bar', x='Customer Segment', y='Return_ind', color='orange',
                      legend=False, figsize=(10, 6))

ax.set_xlabel("X Axis Label")
ax.set_ylabel("Return_ind")
ax.set_title("Top 5 Entries by Return_ind (Bar Plot)")


ax.set_xticklabels(top_entries['Customer Segment'], rotation=45)

plt.show()


# ####  Here we can see that customer segment with highest return index is Young female

# In[75]:


Returnid_Condition=transactions.groupby(['CONDTION'])['Return_ind'].sum().reset_index()

Returnid_Condition


# In[76]:


import plotly.graph_objects as go
bar_color = 'orange'

fig = go.Figure(data=[go.Bar(x=Returnid_Condition['CONDTION'],
                             y=Returnid_Condition['Return_ind'],
                             marker_color=bar_color)])

fig.update_layout(title="Sum of Return_ind by CONDTION (Bar Plot)",
                  xaxis_title="CONDTION",
                  yaxis_title="Sum of Return_ind")

fig.show()


# #### Here we can see that condition with highest return index is new

# In[83]:


Returnid_Productcategory=transactions.groupby(['P_CATEGORY'])['Return_ind'].sum().reset_index()

Returnid_Productcategory


# In[84]:


import plotly.graph_objects as go
fig = go.Figure(data=go.Scatter(x=Returnid_Productcategory['P_CATEGORY'],
                               y=Returnid_Productcategory['Return_ind'],
                               mode='lines+markers',
                               line=dict(color='blue', width=2),
                               marker=dict(color='red', size=8)))

fig.update_layout(title="Sum of Return_ind by P_CATEGORY (Line Plot)",
                  xaxis_title="P_CATEGORY",
                  yaxis_title="Sum of Return_ind")
fig.show()


# #### Here we can see that P_CATEGORY with highest return index is decor

# In[87]:


Returnid_Discount=transactions.groupby(['Discount'])['Return_ind'].sum().reset_index()

Returnid_Discount


# In[86]:


import networkx as nx
import matplotlib.pyplot as plt

G = nx.DiGraph()

for discount in Returnid_Discount['Discount']:
    G.add_node(discount)

for _, row in Returnid_Discount.iterrows():
    G.add_edge(row['Discount'], row['Return_ind'])

plt.figure(figsize=(10, 6))
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=True, node_size=3000, node_color='skyblue', font_size=10, font_weight='bold')
plt.title("Network Graph for Return_ind by Discount")
plt.show()


# ## 4.Create a profile of customers in terms of timing of their order.

# In[88]:


from datetime import datetime

hours=[datetime.strptime(time,"%H:%M:%S").hour for time in data["Time"]]

hours


# In[89]:


hour=pd.DataFrame({'Hour':hours})

print(hours)


# In[90]:


data['Time']=hour['Hour']


# In[94]:


data.head()


# ## 5. Which payment method is providing more discount for customers?

# In[95]:


Max_Discount_Payment_Method=transactions.groupby(['Payment method'])['Discount'].sum().reset_index()  

Max_Discount_Payment_Method


# In[99]:


plt.figure(figsize=(10, 6)) 
plt.bar(Max_Discount_Payment_Method['Payment method'], Max_Discount_Payment_Method['Discount'], color='green')

plt.xlabel("Payment Method")
plt.ylabel("Total Discount")
plt.title("Total Discount by Payment Method (Bar Plot)")

plt.xticks(rotation=45)

plt.show()


# ## 6.Create a profile for high value items vs low value items and relate that wrt to their number of orders.

# In[100]:


mean=data['Selling_price'].mean()
mean


# In[101]:


data['Value_item']=data['Selling_price'].apply(lambda x : 'High_value' if x>=mean else 'Low_value')

data


# In[102]:




data


# ## 7.Do you think if merchant provides more discount then can it will lead to increase in number of orders?

# In[103]:


Merchant_Discount=transactions.groupby(['Merchant_name'])['Coupon_ID'].count().reset_index()

Merchant_Discount


# In[104]:


Merchant_Credit_card=transactions.groupby(['Merchant_name'])['Credit_card'].count().reset_index()

Merchant_Credit_card


# In[105]:


Merchant_Credit_card['Coupon_ID']=Merchant_Discount['Coupon_ID']

Merchant_Credit_card


# here we can see that though we are increasing the discou t we are not getting any change in order from merchant

# In[ ]:





# In[ ]:




