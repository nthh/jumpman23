
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


# In[7]:

get_ipython().magic('matplotlib inline')
import pandas as pd
import seaborn as sns
import json
from urllib.request import urlopen,urlretrieve
from shapely.geometry import shape as Shape, Point

df = pd.read_csv('analyze_me_updated.csv',encoding = "ISO-8859-1")

#boro = urlopen('https://data.cityofnewyork.us/api/geospatial/tqmj-j8zm?method=export&format=GeoJSON').read()


# In[ ]:




# In[ ]:

df.dtypes


# In[71]:

js['features']


# In[62]:

boro[:100]


# In[25]:

import folium


map1 = folium.Map(location=[40.7261, -73.9727], zoom_start=12, tiles='cartodbpositron')
map1.choropleth( data=df.groupby('dropoff_NTA').size(),
    geo_str=urlopen('https://data.cityofnewyork.us/api/geospatial/d3qk-pfyz?method=export&format=GeoJSON').read().decode('utf-8'),
             key_on='properties.ntaname',
             fill_color='PuBuGn', fill_opacity=0.7, line_opacity=0.5,
             legend_name='Dropoffs')
map1


# In[24]:

get_ipython().magic('pinfo map1.choropleth')


# In[78]:

get_ipython().magic('pinfo boro.lstrip')


# In[57]:

get_ipython().magic('pinfo map1.choropleth')


# In[41]:

get_ipython().set_next_input("df.groupby('dropoff_borough').size().reset_index");get_ipython().magic('pinfo reset_index')


# In[ ]:

df.groupby('dropoff_borough').size().reset_index


# In[ ]:

df.groupby('dropoff_borough').size().reset_index


# In[31]:

get_ipython().magic('pinfo map1.choropleth')


# In[29]:

import folium
map1 = folium.Map(location=[40.7261, -73.9727], zoom_start=11.5, tiles='cartodbpositron')
map1


# In[7]:

df.head()


# In[ ]:

flights = sns.load_dataset("flights")
flights = flights.pivot("month", "year", "passengers")


# In[1]:




# In[3]:




# In[152]:




# In[ ]:




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


# In[ ]:




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


# In[29]:




# In[19]:

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


# In[18]:




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


# In[348]:

str(a).replace('[','(').replace(']',')')


# In[346]:


for i in range(len(df[:4])):
    a = []
    for row in df[:4]:
         a.append(df.loc[i,row])
    pickup = "ST_GeomFromText('POINT(%s %s)', 4326)" %(df.iloc[i,11],df.iloc[i,10])
    dropoff = "ST_GeomFromText('POINT(%s %s)', 4326)" %(df.iloc[i,13],df.iloc[i,12])

    
    a.append(pickup + ',' + dropoff)
    print(str(a))

    
            
        

[ [df.loc[i,row] for row in df[:4]] for i in range(len(df[:4]))]  


# In[295]:

df_CensusTracts.head(5)


# In[296]:

df_PUMAtoCensus.head(5)


# In[318]:

df_PUMApop


# In[249]:

df_NTA_to_PUMA.columns = df_NTA_to_PUMA.loc[1]


# In[255]:

df_NTA_to_PUMA.columns[-1] = 'Name'


# In[67]:

## DATA INTEGRITY ##

#550 records where when_the_Jumpman_arrived_at_pickup and _left_pickup is null 
len(df[df.when_the_Jumpman_arrived_at_pickup.isnull()])

print('Null values in each column')
print(df.isnull().sum())
    


# In[131]:

get_ipython().magic('matplotlib inline')
import seaborn as sns
#sns.distplot(pickup_places)


len(pickup_places[pickup_places == 1])
#pickup_places[pickup_places > 100].plot(kind='box')
pd.cut(df.pickup_place, 10)


# In[106]:




# In[101]:

pickup_places[pickup_places == 1].count()


# In[54]:

gis = GIS()
map1 = gis.map()
map1


# In[185]:

us = gis.map('USA',3)

us.add_layer(stamenbasemaps[2])
us


# In[61]:

df.head()


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


# In[92]:




# In[113]:

for idx,row in df[0:200].iterrows():
    drawPoints(ny,row['dropoff_lat'],row['dropoff_lon'],row['delivery_id'],row['item_name'])


# In[100]:

df[df.delivery_id == 1319443].apply(lambda row: getFeatureforPoint(boroughDict, row['dropoff_lon'], row['dropoff_lat']), axis=1)


# In[89]:

from arcgis.gis import GIS
gis = GIS()
ny = gis.map('New York')

stamenbasemaps = gis.content.search("tags:partner_basemap owner:dkensok stamen",
                                    item_type="web map", max_items=3)

#NYCzoning = gis.content.search('title:GeoreferencedNYCZoningMaps ')
#ny.add_layer(NYCzoning[0])
#ny.add_layer(NYCzoning[0].layers[0])
ny


# In[142]:

#search_result[0].layers[0]
map2 = a.MapView(stamenbasemaps[1],'NY')
map2


#location = gis.tools.geocoder.find_best_match('New York City')


# In[112]:

ny.center = [40.7261, -73.9727]


ny.zoom = 11


# In[147]:

lat = []
lon = []
mag = []
sr = df.groupby(['pickup_lat','pickup_lon']).size()

for idx, row in sr.items():
    lat.append(idx[0])
    lon.append(idx[1])
    mag.append(row)
    
lat2 = []
lon2 = []
mag2 = []
sr2 = df.groupby(['dropoff_lat','dropoff_lon']).size()

for idx, row in sr2.items():
    lat2.append(idx[0])
    lon2.append(idx[1])
    mag2.append(row)


# In[151]:

import folium
district_geo = r'sfpddistricts.geojson'

# calculating total number of incidents per district
map1 = folium.Map(location=[40.7261, -73.9727], zoom_start=12)

# crimedata2 = pd.DataFrame(crimedata['PdDistrict'].value_counts().astype(float))

# crimedata2.to_json('crimeagg.json')

# crimedata2 = crimedata2.reset_index()

# crimedata2.columns = ['District', 'Number']

  

# # creation of the choropleth

# map1 = folium.Map(location=SF_COORDINATES, zoom_start=12)

# map1.geo_json(geo_path = district_geo,

#               data_out = 'crimeagg.json',

#               data = crimedata2,

#               columns = ['District', 'Number'],

#               key_on = 'feature.properties.DISTRICT',

#               fill_color = 'YlOrRd',

#               fill_opacity = 0.7,

#               line_opacity = 0.2,

#               legend_name = 'Number of incidents per district')

               


from folium import plugins

#map1.add_child(plugins.HeatMap(zip(lat, lon, mag), radius = 10,gradient = {0.4: 'skyblue', 0.65: 'deepskyblue', 1: 'royalblue'}))
map1.add_child(plugins.HeatMap(zip(lat2, lon2, mag2), radius = 10,gradient = {0.4: 'lightgreen', 0.65: 'mediumseagreen', 1: 'green'}))



display(map1)


# In[104]:

ny


# In[101]:

from arcgis import geocoding
from arcgis.widgets import MapView

    
newspaper = gis.content.search('title:Stamen toner type:Web Map')[0]
ny = MapView(item=newspaper)
#showargs(arcgis.widgets.MapView


# In[212]:

def showargs(function):
    def inner(*args, **kwargs):
        return function((args, kwargs), *args, **kwargs)
    return inner    


# In[45]:


map1 = gis.map('USA',3)
search_result = gis.content.search('title:USA freeway system AND owner:esri',
                                  item_type = 'Feature Layer')
search_result

freeway_item = search_result[0]
#map1.add_layer(freeway_item)

freeway_feature_layer = freeway_item.layers[0]
map1.add_layer(freeway_feature_layer)

map1.center = [34, -118]
map1.zoom = 2
map1


# In[33]:

map1


# In[31]:

list(freeway_item.layers)


# In[5]:

df.pickup_lat.max()


# In[2]:

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import matplotlib.cm
 
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize


# In[13]:

df.pickup_lon.m()


# In[74]:

pickups


# In[20]:


fig, ax = plt.subplots(figsize=(10,20))
m = Basemap(resolution='i', # c, l, i, h, f or None
            projection='merc',
            lat_0=40.7261, lon_0=-73.9727,
            llcrnrlon=-74.02, llcrnrlat= 40.60, urcrnrlon=-73.90, urcrnrlat=40.88)

pickups = df.groupby(['pickup_lat','pickup_lon']).size()
for i,cnt in enumerate(pickups):
    lat, lng = pickups.axes[0][i]
    size = (cnt/20) ** 2 + 3
    m.plot(lng, lat, 'o', markersize=size, color='#444444', alpha=0.8, latlon = True)

m.drawmapboundary(fill_color='#99ffff')
m.fillcontinents(color='#cc9966',lake_color='#99ffff')


m.readshapefile('./NYC_boundaries/geo_export_9f38c1bb-16ff-437f-95a6-acb0071044de', 'areas')
m.scatter(40.7261,-73.9727,300,marker='o',color='g',latlon=True)

m


# In[5]:

import gmaps
import gmaps.datasets

# load a Numpy array of (latitude, longitude) pairs
locations = gmaps.datasets.load_dataset("taxi_rides")



# In[39]:




# In[37]:

coords = [coords[0],coords[0],coords[0],coords[0],coords[0],coords[1],coords[1],coords[2]]

coords[0]


# In[148]:




# In[67]:


placeList = urllib2.urlopen("http://partners.api.skyscanner.net/apiservices/geo/v1.0?apiKey=ja434940493145398250481019156214").read()
import json
parsed_json = json.loads(placeList)


# In[97]:

json_normalize(parsed_json['Continents'])


# In[98]:

import pandas as pd
from pandas.io.json import json_normalize

json_normalize(parsed_json['Continents'][1]['Countries'])


# In[152]:

import gmaps
gmaps.configure(api_key="AIzaSyDqUF-lwRLvvioTJ0lJLPWCWtSLZt8uw7U") # Your Google API key

bike_dropoffs = list(df[['dropoff_lat','dropoff_lon']][df.vehicle_type == 'bicycle'].itertuples(index=False))
car_dropoffs = list(df[['dropoff_lat','dropoff_lon']][df.vehicle_type == 'car'].itertuples(index=False))


# In[154]:


m = gmaps.Map()


#def makeHeatmapLayer(data, gradient =  )
coords = list(df[['dropoff_lat','dropoff_lon']].itertuples(index=False))
bikedrop_layer = gmaps.Heatmap(data=bike_dropoffs, max_intensity=10)
#bikedrop_layer.gradient = [ 'white',  'silver', 'gray' ]
#bikedrop_layer.gradient = [ 'clear',  'yellow', 'red' ]
bikedrop_layer.opacity = 0.3


m.add_layer(bikedrop_layer)
m


# In[55]:

bikedrop_layer.


# In[35]:




# In[26]:




# In[20]:

[(df['pickup_lat'],df['pickup_lon'])][0]


# In[ ]:

gm = gmaps.Heatmap


# In[19]:

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

fig, ax = plt.subplots(figsize=(10,20))

m = Basemap(resolution='f', # c, l, i, h, f or None
            projection='merc',
            lat_0=54.5, lon_0=-4.36,
            llcrnrlon=-6., llcrnrlat= 49.5, urcrnrlon=2., urcrnrlat=55.2) 

m.drawmapboundary(fill_color='#46bcec')
m.fillcontinents(color='#f2f2f2',lake_color='#46bcec')
m.drawcoastlines()

def plot_area(pos):
    count = new_areas.loc[new_areas.pos == pos]['count']
    x, y = m(pos[1], pos[0])
    size = (count/1000) ** 2 + 3
    m.plot(y, x, 'o', markersize=size, color='#444444', alpha=0.8,latlon)
    
new_areas.pos.apply(plot_area)


# In[361]:

df.groupby('delivery_id').size() > 1


# In[115]:

df.groupby(['pickup_lat','pickup_lon']).size()[:4]


for i,cnt in enumerate(pickups[:100]):
    lat, lng = pickups.axes[0][i]
    size = (cnt/10) ** 2 + 3
    m.scatter(x=lng, y=lat, marker = 'o',  color='#444444', latlon=True)


# In[325]:

from urllib.request import urlopen

url = "https://data.cityofnewyork.us/resource/7t3b-ywvw.json"

boroughs = urlopen(url).read().decode('utf-8')
js = json.loads(boroughs)


# In[360]:

import geopandas as gpd

gdf = gpd.GeoDataFrame(js[0])


# In[368]:

b = shape(js[0]['the_geom'])
b.simplify()


# In[362]:

gdf.head()


# In[370]:

get_ipython().run_cell_magic('time', '', "getValueForPointJson(boroughs,'boro_name',-74.002747,40.752073) ")


# In[380]:

get_ipython().run_cell_magic('time', '', 'getValueForPoint(boroughBoundaries,0,-74.002747,40.752073) ')


# In[434]:

from shapely.geometry import shape as Shape, Point


shape(js[0]['the_geom'])


# In[381]:

polygon = shape(js[0]['the_geom'])


# In[397]:

js[0]['boro_name']


# In[444]:

get_ipython().magic('time')



# In[422]:

def getFeatureforPoint(shapeDict,lon,lat,feature='boro_name'):
    
    for 
    
    point = Point(lon,lat)
    for key, shp in shapeDict.items():
        if shp.contains(point):
            return(key)


# In[446]:

df[700:705].apply(lambda row: getFeatureforPoint(boroughs, row['pickup_lon'], row['pickup_lat']), axis=1)


# In[443]:


df[700:705].apply(lambda row: getValueForPointJson(js, 'boro_name',row['pickup_lon'], row['pickup_lat']), axis=1)


# In[ ]:

boroughs.


# In[408]:

get_ipython().magic('time')
point = Point(-74.002747,40.752073)
for name,shp in boroughs.items():
    if shp.contains(point):
            print( 'Found containing polygon:', name)


# In[ ]:




# In[383]:

for boro in js:
    print(boro['boro_name'])


# In[429]:

def getValueForPointJson(geoJson,field,lon,lat):
    
    

    point = Point(lon, lat)
    
    for shp in js:
        polygon = shape(shp['the_geom'])
        if polygon.contains(point):
            return(shp[field])
        

    
#     for shpRecord in r.iterShapeRecords():
#         if shape(shpRecord.shape).contains(Point(lon, lat)):
#             return shpRecord.record[valIdx]


# In[312]:

js


# In[298]:




# In[311]:

get_ipython().run_cell_magic('time', '', "import json\ndf[0:1].apply(lambda row: getValueForPointJson(boroughs, 0,\n                                                              row['pickup_lon'], row['pickup_lat']), axis=1)")


# In[301]:

js


# In[285]:

get_ipython().run_cell_magic('time', '', 'for row in df[0:2].iterrows():\n    print(getValueForPoint(boroughBoundaries,0,row[1].dropoff_lon,row[1].dropoff_lat))\n    ')


# In[ ]:




# In[283]:

row[1].dropoff_lat


# In[270]:

df[0:2][['pickup_lon', 'pickup_lat']]


# In[256]:

len(df)


# In[224]:



# read your shapefile
r = shapefile.Reader("./NYC_Boundaries/geo_export_aafeafbb-903e-43a3-b9be-b75895259518.shp")

# get the shapes
shapes = r.shapeRecords()

# build a shapely polygon from your shape
#polygon = shape(shapes[0])    

def check(polygon,lon, lat):
    # build a shapely point from your geopoint
    polygon = shape(polygon)
    point = Point(lon, lat)
    
    # the contains function does exactly what you want
    return polygon.contains(point)


# In[164]:

check(-74.002747,40.752073)


# In[241]:




# In[236]:

sf = 


# In[214]:

for bor in boroughDict:
    print(bor,check(boroughDict[bor],-74.002747,40.752073))


# In[194]:

boroughDict = {}
for i,borough in enumerate(list(r.iterRecords())):
    boroughDict[borough[0]] = shapes[i]


# In[205]:

a = shape(boroughDict['Bronx'])


# In[155]:

import shapely


# In[138]:

print('Average wait time:',df.order_wait_time.mean().seconds//60)


# In[116]:




# In[99]:




# In[122]:

df.dtypes


# In[95]:

df.


# In[88]:

df.dtypes


# In[81]:

for row in df.when_the_Jumpman_arrived_at_pickup[0:10]:
    print(row,type(row))
    


# In[60]:

df.dtypes


# In[47]:

datetime.strptime('2014-10-16 22:48:23.091253','%Y-%m-%d %H:%M:%S.%f')


# In[42]:

df.when_the_Jumpman_arrived_at_dropoff - df.when_the_delivery_started 


# In[41]:

df.dtypes

