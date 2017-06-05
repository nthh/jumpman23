
# coding: utf-8

# In[4]:

get_ipython().magic('matplotlib inline')
import pandas as pd
import seaborn as sns
import json
from urllib.request import urlopen,urlretrieve
from shapely.geometry import shape as Shape, Point


# Major analytical tasks on the city level will include monitoring and interpreting data on local order volume, 
# local fleet size/growth, 
# fleet quality and engagement,
# local customer growth, 
# merchant performance, 
# customer, 
# fleet and merchant feedback, 
# market settings and density & zoning. 

#Read to DataFrame and date/time formatting
df = pd.read_csv('analyze_me.csv',
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



#see if it takes longer to order for certain vendors


# In[145]:

df.head()


# In[7]:

get_ipython().magic('matplotlib inline')
import pandas as pd
import seaborn as sns
import json
from urllib.request import urlopen,urlretrieve
from shapely.geometry import shape as Shape, Point

df = pd.read_csv('analyze_me_updated.csv',encoding = "ISO-8859-1")

#boro = urlopen('https://data.cityofnewyork.us/api/geospatial/tqmj-j8zm?method=export&format=GeoJSON').read()


# In[43]:

df.groupby(['dropoff_lon','dropoff_lat']).size().transform(lambda x: (x ) / x.max())


# In[25]:

import folium


map1 = folium.Map(location=[40.7261, -73.9727], zoom_start=12, tiles='cartodbpositron')
map1.choropleth( data=df.groupby('dropoff_NTA').size(),
    geo_str=urlopen('https://data.cityofnewyork.us/api/geospatial/d3qk-pfyz?method=export&format=GeoJSON').read().decode('utf-8'),
             key_on='properties.ntaname',
             fill_color='PuBuGn', fill_opacity=0.7, line_opacity=0.5,
             legend_name='Dropoffs')
map1


# In[29]:

import folium
map1 = folium.Map(location=[40.7261, -73.9727], zoom_start=11.5, tiles='cartodbpositron')
map1


# In[7]:

df.head()


# In[ ]:

flights = sns.load_dataset("flights")
flights = flights.pivot("month", "year", "passengers")


# In[43]:

df.head()


# # Local Order Volume
# 

# In[52]:

for field, desc in {'delivery_id':'orders','pickup_place':'pickup places',
        'customer_id':'customers','jumpman_id':'Jumpmen','item_name':'items','place_category':'place categories'}.items():
    print('Number of %s:'% desc,len(df[field].unique()))


# In[ ]:

print('order')


# In[20]:

print('Cross-borough deliveries')
df[df.dropoff_borough != df.pickup_borough].groupby(['pickup_borough','dropoff_borough']).size()     .sort_values(ascending=False).reset_index(name='count')


# In[21]:

print('Cross-neighborhood deliveries')
df[df.dropoff_NTA != df.pickup_NTA].groupby(['pickup_NTA','dropoff_NTA']).size()     .sort_values(ascending=False).reset_index(name='count').head(10)


# In[452]:

df.groupby('dropoff_borough').size()


# In[35]:

customers = df.groupby('customer_id').size()
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


# In[ ]:

import numpy as np

sns.distplot(popular_places2.tolist())


# In[5]:

popular_places = df.groupby('place_category').size().sort_values(ascending=False).head(10)

sns.plt.suptitle('Most popular place categories')
sns.barplot(y=popular_places.axes[0],x=popular_places.values)


# In[17]:

vehicles = df.groupby('vehicle_type').size()

sns.plt.suptitle('Orders by Vehicle Type')
sns.barplot(x=vehicles.values,y=vehicles.axes[0])


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




# In[324]:

PUMApopulation.head(5)


# In[295]:

df_CensusTracts.head(5)


# In[296]:

df_PUMAtoCensus.head(5)


# In[67]:

## DATA INTEGRITY ##

#550 records where when_the_Jumpman_arrived_at_pickup and _left_pickup is null 
len(df[df.when_the_Jumpman_arrived_at_pickup.isnull()])

print('Null values in each column')
print(df.isnull().sum())
    


# In[95]:


def drawPoints(mapObj,lat,lon,title,content):

    mapObj.draw(shape=[lat, lon], attributes={"title":"Search Location", "content":"Predicted crash location"},
           symbol = {
    "type": "esriSMS",
     "style": "esriSMSCircle",
     "color": [76,115,0,255],
     "size": 3,
     "angle": 0,
     "xoffset": 0,
     "yoffset": 0,
     
    }, popup = {'title':title,'content':content})


# In[112]:

ny.center = [40.7261, -73.9727]


ny.zoom = 11

