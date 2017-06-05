from flask import Flask, render_template, jsonify, send_from_directory
import pandas as pd
import json

app = Flask(__name__)

df = pd.read_csv('./data/analyze_me_updated.csv',encoding = "ISO-8859-1")


@app.route("/jupyter")
def jupyter():
    return app.send_static_file('jm23-nyc.html')
	
@app.route("/")	
def index():

	vehicle_types = df.vehicle_type.unique();
	place_categories = df.groupby('place_category').size().sort_values(ascending=False).head(15).axes[0].tolist()

	return render_template('index.html', vehicle_types = vehicle_types, place_categories = place_categories)
	

@app.route('/<coordType>/<field>/<name>')
def generateHeatmapData(coordType, field, name):
	
	s = df[df[field] == name].groupby([ (coordType + '_lat'),(coordType + '_lon')]).size()
	
	data = [ [a[0],a[1]] for a,b in s.items() ]
	return jsonify({"data": data})
	

@app.route('/dropoff_heatmap')
def dropoff_heatmap():
	s = df.groupby(['dropoff_lat','dropoff_lon']).size().transform(lambda x: (x ) / x.max())
	data = [ [a[0],a[1]] for a,b in s.items() ]
	return jsonify({"data": data})
	
@app.route('/pickup_heatmap')
def pickup_heatmap():
	s = df.groupby(['pickup_lat','pickup_lon']).size().transform(lambda x: (x ) / x.max())
	data = [ [a[0],a[1] ] for a,b in s.items() ]
	return jsonify({"data": data})


	
	
		


def df_to_geojson(df, properties, lat='lat', lon='long', z='elev'):
    geojson = {'type':'FeatureCollection', 'features':[]}
    for _, row in df.iterrows():
        feature = {'type':'Feature',
                   'properties':{},
                   'geometry':{'type':'Point','coordinates':[]}}
        feature['geometry']['coordinates'] = [row[lon],row[lat]]
        for prop in properties:
            feature['properties'][prop] = row[prop]
        geojson['features'].append(feature)
    return geojson



	
#Access to stored data
@app.route('/data/<string:boundaryData>')
def loadFeatures(boundaryData):
    return send_from_directory('data', boundaryData)
	
@app.route("/pickups_Boroughs")
def pickups_Boroughs():
	
	
	with open('data/boroughBoundaries.geojson') as f:
		data = json.load(f)
		
	for feature in data['features']:
		try:
			feature['properties']['pickups'] = df.groupby('pickup_borough').size()[feature['properties']['boro_name']]

		except KeyError:	
			feature['properties']['pickups'] = 0

	
	return jsonify(data);
	


@app.route("/pickups_NTA/")	
@app.route("/pickups_NTA/<field>/<name>")
def pickups_NTA(field = None,name = None):
	
	
	with open('data/ntaBoundaries.geojson') as f:
		data = json.load(f)
		
	for feature in data['features']:
		
		try:
			if field is None:
				feature['properties']['values'] = int(df.groupby('pickup_NTA').size()[feature['properties']['ntaname']])
			else:
				feature['properties']['values'] = int(df[df[field] == name].groupby('pickup_NTA').size()[feature['properties']['ntaname']])
				

		except KeyError:
			feature['properties']['values'] = int(0)

	
	return jsonify(data);

if __name__ == '__main__':
    # Bind to PORT if defined, otherwise default to 5000.
	port = int(os.environ.get('PORT', 5000))
	app.run(host='0.0.0.0', port=port)
