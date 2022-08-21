#from application import app
import auth
import feature_get
import pandas as pd
import requests
import flask
from flask import Flask 
from flask import render_template

app= Flask(__name__)
app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba241'

"""d=feature_get.get_playlist()
df = pd.DataFrame(d)
songDF = feature_get.drop_duplicates(df)
songDF = feature_get.select_cols(songDF)

float_columns = [x for x in d if x in ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']]
complete_feature_set = feature_get.create_feature_set(songDF, float_columns)"""

@app.route("/")
def home():
   #render the home page
   return render_template('home.html')


@app.route('/recommend', methods=['POST'])
def recommend():
   #requesting the URL form the HTML form
   URL = flask.request.form['URL']
   #using the extract function to get a features dataframe
   df = feature_get.get_playlist(URL)
   #retrieve the results and get as many recommendations as the user requested
   top40 = feature_get.recommend_from_playlist(df)
   number_of_recs = 9
   image_urls=[]
   my_songs = []
   for i in range(number_of_recs):
      #my_songs.append([str(top40.iloc[i,1]) + ' - '+ '"'+str(top40.iloc[i,4])+'"', "https://open.spotify.com/track/"+ str(top40['id'])])
      my_songs.append([str(top40.iloc[i]['track_name']), "https://open.spotify.com/track/"+ str(top40.iloc[i]['id'])])
      image_urls.append(feature_get.auth.sp.track(top40['id'][i])['album']['images'][0]['url'])
      #images.append([i['uri'].split(':')[2]] = i['images'][0]['url']
   return render_template('results.html', len=len(my_songs), songs= my_songs, images=image_urls)


if __name__ == '__main__':
	app.run(debug=False)