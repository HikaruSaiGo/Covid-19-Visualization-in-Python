#!/usr/bin/env python
# coding: utf-8

# # Covid-19 Visualization

# ### Import Package

# In[12]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from datascience import *
import warnings 
warnings.filterwarnings("ignore")


# ### Import Dataset

# In[2]:


confirmed = pd.read_csv('time_series_covid19_confirmed_global_narrow.csv', skiprows = [1], dtype = {'Value': int}) #remove the 2nd row

confirmed['Status'] = 'Confirmed'

confirmed


# In[3]:


death = pd.read_csv('time_series_covid19_deaths_global_narrow.csv', skiprows = [1], dtype = {'Value': int})

death['Status'] = 'Death'

death


# ### Merge Dataset

# In[4]:


total = pd.concat([confirmed, death], ignore_index = True)

total


# ### Total Covid-19 Cases All Over the World Until 2/2/2022

# In[5]:


tot_num = total.query("Date == '2022-02-02'")[["Status", "Value"]].groupby(["Status"], as_index = False).sum()

tot_num


# In[13]:


fig, ax = plt.subplots(figsize = (12, 8))  # using this when plt.figure(figsize = (12, 8)) doesn't work
tot_num.plot(kind = "barh", x = "Status", legend = False, ax = ax)
plt.title("Total Confirmed Cases vs. Death Cases Worldwide", y = 1.05, x = 0.5)
plt.xlabel("")
plt.ylabel("Status")
plt.ticklabel_format(axis = "x", style = "plain") # plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.show()


# ### Total Confirmed & Death Cases for Each Country/Region

# In[9]:


df = total.query("Date == '2022-02-02'")[["Date", "Country/Region", "Status", "Value"]].groupby(["Status", "Country/Region"], as_index = False).sum()

df


# ### Top 10 Countries/Regions with Most Confirmed Cases 

# In[10]:


con_top = df.query("Status == 'Confirmed'").sort_values(by = ['Value'], ascending = False).head(10)

con_top


# In[14]:


sns.catplot(x = "Value", y = "Country/Region", data = con_top, kind = "bar", height = 8, aspect = 12/8, palette = "RdYlBu")
plt.title("Top 10 Countries with Most Confirmed Cases", y = 1.05, x = 0.5)
plt.xlabel("Cases")
plt.ticklabel_format(axis = "x", style = "plain") 
plt.show()


# ### Top 10 Countries/Regions with Most Death Cases 

# In[15]:


dea_top = df.query("Status == 'Death'").sort_values(by = ['Value'], ascending = False).head(10)

dea_top


# In[16]:


sns.catplot(x = "Value", y = "Country/Region", data = dea_top, kind = "bar", height = 8, aspect = 12/8, palette = "PRGn")
plt.title("Top 10 Countries with Most Death Cases", y = 1.05, x = 0.5)
plt.xlabel("Cases")
plt.ticklabel_format(axis = "x", style = "plain") 
plt.show()


# In[17]:


#df.groupby('Status', as_index = False).apply(lambda x: x.sort_values('Value', ascending = False).head(10))

#df.sort_values('Value', ascending = False).groupby("Status", as_index = False).head(10).reset_index()


# ### List of Top 10 Countries/Regions with Most Confirmed & Death Cases

# In[18]:


nam = df.sort_values('Value', ascending = False).groupby("Status", as_index = False).head(10)["Country/Region"].values

nam 


# In[19]:


df_top = df[df["Country/Region"].isin(nam)].sort_values('Value', ascending = False)

df_top


# ### Convert Long Table into Wide Table

# In[20]:


pd.pivot(df_top, index = "Country/Region", columns = "Status", values = "Value").rename_axis(columns = None).reset_index()


# ### Convert a Pandas DataFrame into a Table

# In[21]:


new = Table.from_df(pd.pivot(df_top, index = "Country/Region", columns = "Status", values = "Value").rename_axis(columns = None).reset_index())

new


# In[22]:


new.sort("Confirmed", descending = True).barh("Country/Region", height = 8, width = 12)
plt.title("Confirmed Cases vs. Death Cases", y = 1.05, x = 0.5)
plt.xlabel("Cases")
plt.ticklabel_format(axis = "x", style = "plain") # plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))    
plt.show()


# ### Visualization on Map

# In[23]:


loc = total.drop_duplicates(subset = "Country/Region")[["Country/Region", "Lat", "Long"]]

loc


# In[24]:


newloc = pd.merge(df, loc, how = 'inner', on = ['Country/Region'])

newloc


# In[25]:


import geopandas
cov19data = geopandas.GeoDataFrame(newloc, geometry = geopandas.points_from_xy(newloc.Long, newloc.Lat))
world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
world_cov19 = geopandas.sjoin(world, cov19data, how = "inner", op = 'intersects')

world_cov19


# ### Covid-19 Confirmed Cases

# In[26]:


fig, ax = plt.subplots(1, 1, figsize = (15,10))
world_cov19.query("Status == 'Confirmed'").plot(column = 'Value', ax = ax, cmap = 'Reds', scheme = 'NaturalBreaks', k = 6, legend = True,
                legend_kwds = {'loc':'lower left','title':'Covid-19 Confirmed Cases'})
plt.show()


# In[27]:


fig, ax = plt.subplots(1, 1, figsize = (15,10))

world_cov19.query("Status == 'Confirmed'").plot(column = 'Value', cmap = 'OrRd', scheme = 'quantiles', ax = ax, legend = True,
                                               legend_kwds = {'loc':'lower left','title':'Covid-19 Confirmed Cases'})

plt.show()


# ### Covid-19 Death Cases

# In[28]:


fig, ax = plt.subplots(1, 1, figsize = (15,10))
world_cov19.query("Status == 'Death'").plot(column = 'Value', ax = ax, cmap = 'Blues', scheme = 'NaturalBreaks', k = 6, legend = True,
                legend_kwds = {'loc':'lower left','title':'Covid-19 Death Cases'})
plt.show()


# In[29]:


fig, ax = plt.subplots(1, 1, figsize = (15,10))
world_cov19.query("Status == 'Death'").plot(column = 'Value', ax = ax, legend = True)
plt.show()


# ### Confirmed & Death Cases Every Day

# In[30]:


dtotal = total[["Date", "Country/Region", "Status", "Value"]].groupby(["Date","Status"], as_index = False).sum()

dtotal


# In[31]:


import plotly.express as px

# color pallette
cnf, dth = '#2131bf', '#ff2e63'

fig = px.area(dtotal, x = "Date", y = "Value", color = 'Status', height = 600,
             title = 'Cases Over Time', color_discrete_sequence = [cnf,dth])
fig.show()


# In[ ]:




