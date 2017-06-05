
# coding: utf-8

# # Jumpman 23 NYC Analysis
# 
# ### Nathan Hearnsberger 
# #### nhearnsberger@gmail.com
# 
# 
# As can be seen to the right, New York City's orders are highly concentrated in Lower Manhattan. 
# 
# Click the "Pickups" or "Dropoffs" button and then a vehicle type or place category to view a heatmap of the order quantity. 
# 

# In[84]:

get_ipython().magic('matplotlib inline')
import pandas as pd
import seaborn as sns
import json
from urllib.request import urlopen,urlretrieve
from shapely.geometry import shape as Shape, Point


#Read to DataFrame and date/time formatting
df = pd.read_csv('./data/analyze_me.csv',
                 parse_dates= ['when_the_delivery_started',
                              'when_the_Jumpman_arrived_at_pickup',
                              'when_the_Jumpman_left_pickup',
                              'when_the_Jumpman_arrived_at_dropoff'],
                 infer_datetime_format = True,
                 converters = {'how_long_it_took_to_order':pd.to_timedelta})

#Add fields for pickup/order wait times
df['pickup_wait_time'] = (df.when_the_Jumpman_left_pickup - df.when_the_Jumpman_arrived_at_pickup)   
df['order_wait_time'] = (df.when_the_Jumpman_arrived_at_dropoff - df.when_the_delivery_started)   


#Returns feature for coordinates using dict of feature:Shape
def getFeatureforPoint(shapeDict,lon,lat):
    point = Point(lon,lat)
    for feature, shape in shapeDict.items():
        if shape.contains(point):
            return(feature)

#Geodata from NYC OpenData 
boroughBoundaries = urlopen('https://data.cityofnewyork.us/resource/7t3b-ywvw.json').read().decode('utf-8')
boroughDict = { borough['boro_name']: Shape(borough['the_geom']) for borough in json.loads(boroughBoundaries)}

#NTA (Neighbordhood Tabulation Area) Boundaries 
ntaBoundaries = urlopen('https://data.cityofnewyork.us/resource/93vf-i5bz.json').read().decode('utf-8')
ntaDict = { nta['ntaname']: Shape(nta['the_geom']) for nta in json.loads(ntaBoundaries)}

#Add borough and NTA for each pickup/dropoff location
df['pickup_borough'] = df.apply(lambda row: getFeatureforPoint(boroughDict,  row['pickup_lon'], row['pickup_lat']), axis=1)
df['dropoff_borough'] = df.apply(lambda row: getFeatureforPoint(boroughDict, row['dropoff_lon'], row['dropoff_lat']), axis=1)
df['pickup_NTA'] = df.apply(lambda row: getFeatureforPoint(ntaDict,  row['pickup_lon'], row['pickup_lat']), axis=1)
df['dropoff_NTA'] = df.apply(lambda row: getFeatureforPoint(ntaDict, row['dropoff_lon'], row['dropoff_lat']), axis=1)


# In[46]:

df.head()


# # Local Order Volume
# 

# In[45]:

for field, desc in {'delivery_id':'orders','pickup_place':'pickup places',
        'customer_id':'customers','jumpman_id':'delivers','item_name':'items','place_category':'place categories'}.items():
    print('Number of %s:'% desc,len(df[field].unique()))


# In[48]:

df.groupby('dropoff_borough').size().reset_index(name='count')


# In[60]:

print('Dropoffs by neighborhood')
df.groupby('dropoff_NTA').size().sort_values(ascending=False).reset_index(name='count').head(20)


# In[20]:

print('Cross-borough deliveries')
df[df.dropoff_borough != df.pickup_borough].groupby(['pickup_borough','dropoff_borough']).size()     .sort_values(ascending=False).reset_index(name='count')


# In[21]:

print('Cross-neighborhood deliveries')
df[df.dropoff_NTA != df.pickup_NTA].groupby(['pickup_NTA','dropoff_NTA']).size()     .sort_values(ascending=False).reset_index(name='count').head(10)


# In[132]:

print('Deliveries by day')
df.groupby(df.when_the_delivery_started.dt.weekday_name).size().sort_values(ascending=False).reset_index(name='count')


# In[49]:

list(df)


# In[81]:

df.groupby(['day','hour']).size()


# In[134]:





# In[105]:

list(df)


# In[154]:

df['day'] = df.when_the_delivery_started.dt.weekday_name
df['hour'] = df.when_the_delivery_started.dt.hour
d = df.groupby(['day','hour']).size().reset_index(name='count').pivot('day','hour','count')
d.index = pd.CategoricalIndex(d.index, categories= ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"])
d.sort_index(level=0, inplace=True)
sns.heatmap(d,linewidths=.004,cmap='YlOrRd')


# In[61]:

print('Most popular pickup places')
pickup_places = df.groupby('pickup_place').size()
pickup_places.sort_values(ascending=False).head(20).reset_index(name='count')


# In[32]:

print('Places with %s order: %s'%(1,len(pickup_places[pickup_places == 1])))
for x in [[2,5],[6,20],[20,max(pickup_places)]]:
    print('Places with %s to %s orders: %s'%(x[0],x[1],len(pickup_places[pickup_places.between(x[0],x[1])])) )  


# In[33]:

print('Customers with %s order: %s'%(1,len(customers[customers == 1])))
for x in [[2,5],[6,20],[20,max(customers)]]:
    print('Customers with %s to %s orders: %s'%(x[0],x[1],len(customers[customers.between(x[0],x[1])])) )  


# In[62]:

popular_places = df.groupby('place_category').size().sort_values(ascending=False).head(10)

sns.plt.suptitle('Most popular place categories')
sns.barplot(y=popular_places.axes[0],x=popular_places.values)


# In[17]:

vehicles = df.groupby('vehicle_type').size()

sns.plt.suptitle('Orders by Vehicle Type')
sns.barplot(x=vehicles.values,y=vehicles.axes[0])


# ## Merchant performance
# 
# Merchants for which users take longer to order may have menus that are not optimized for efficient ordering. These merchants can be proactively engaged to prevent the loss of merchants and/or users.
# 
# The same analysis can be done for merchants that have a long wait time. Time spent waiting for the food to be ready prevents the Jumpan from making other deliveries.

# In[96]:

print('Longest Time to Order')
df.groupby('pickup_place').how_long_it_took_to_order.describe()['mean'].sort_values(ascending=False).head(15)


# In[97]:

print('Greatest Pickup Wait Times')
df.groupby('pickup_place').pickup_wait_time.describe()['mean'].sort_values(ascending=False).head(15)


# ## Data quality
# 
# From a general analysis, the data provided appears to have little data quality issues, but there is a high amount of data with missing item information and missing times for when the Jumpan arrived and left the pickup site.
# 
# In addition, there are 16 rows which have exact duplicates. Root cause analysis should have to be performed to see if this is a legitimate issue or actual orders with the same item multiple times.

# In[144]:

print('Null values in each column')
df.isnull().sum()


# In[119]:

print('Complete duplicate records')
df[df.duplicated()]


# ## Future Analysis
# 
# By comparing the populations of neighborhoods against deliveries, underserved areas could be determined. This would be able to assist marketing and staffing efforts to improve service to these areas.

# In[319]:


## Data from NYC OpenData (https://data.cityofnewyork.us/) 

# PUMA to Census Tract Mapping
df_PUMAtoCensus = pd.read_excel('nyc2010census_tabulation_equiv.xlsx',skiprows=3)
#df_PUMAtoCensus.columns = df_PUMAtoCensus.loc[1]

# 2010 Population by Census Tract
df_CensusTracts = pd.read_csv('New_York_City_Population_By_Census_Tracts.csv')

#Joining the two tables to get population by PUMA
PUMApopulation = pd.merge(df_CensusTracts, 
         df_PUMAtoCensus[['2010 Census Tract','PUMA']], 
         left_on = 'Census Tract', 
         right_on = '2010 Census Tract').groupby('PUMA').sum()['Population']




# In[295]:

df_CensusTracts.head(5)


# In[296]:

df_PUMAtoCensus.head(5)

