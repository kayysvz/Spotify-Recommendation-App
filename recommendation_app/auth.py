import spotipy
import spotipy.util
from spotipy.oauth2 import SpotifyClientCredentials
import requests
import base64
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

#Authentication - without user
def authenticate():
    with open(r"C:\Users\kayys\PracticalDataScience-ENCA\secret.txt", 'r') as f:
        secret_ls = f.readlines()
        cid = secret_ls[0][:-1]
        secret = secret_ls[1]
        f.close()


    cid=cid.strip()
    secret=secret.strip()

    username='kayysvz@gmail.com'
    scope='user-library-read'

    #cid = cid.encode('ascii')
    #base64cid = base64.b64encode(cid)
    #secret = secret.encode('ascii')
    #base64secret = base64.b64encode(secret)

    token = spotipy.util.prompt_for_user_token(username,
                            scope,
                            client_id=cid,
                            client_secret=secret,
                            redirect_uri='http://localhost:8888/callback')

    return(spotipy.Spotify(auth=token))
sp=authenticate()

def get_playlist():
    #if token:
    #client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
    # = SpotifyClientCredentials(client_id='64fd2a7e6edd467cbe8f2a8e099a4eab', client_secret='b4246dc052dd4fcda242b37c08e3ec07')
    #sp = spotipy.Spotify(client_credentials_manager = client_credentials_manager)


    playlist_link = "https://open.spotify.com/playlist/37i9dQZEVXbNG2KDcFcKOF?si=1333723a6eff4b7f"
    playlist_URI = playlist_link.split("/")[-1].split("?")[0]
    track_uris = [x["track"]["uri"] for x in sp.playlist_tracks(playlist_URI)["items"]]
    
    column_names = ['artist_name','id','track_name','danceability', 'energy', 'key', 'loudness', 'mode',
    'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', "artist_pop", "track_pop",
    'track_uri', 'track_name', 'artist_uri', 'album', 'artist_info', 'artist_genres', 'type', 'uri', 'track_href', 'analysis_url',
    'duration_ms', 'time_signature']

    song_features = {k: [] for k in column_names}
    
    
    for track in sp.playlist_tracks(playlist_URI)["items"]:
        #URI
        track_uri = track["track"]["uri"]
        song_features['track_uri'].append(track["track"]["uri"])
        
        #Track name
        song_features["track_name"].append(track["track"]["name"])
        
        #Main Artist
        artist_uri = track["track"]["artists"][0]["uri"]
        song_features["artist_uri"].append(artist_uri)
        artist_info = sp.artist(artist_uri)
        song_features["artist_info"].append(artist_info)
        
        #Name, popularity, genre
        song_features["artist_name"].append(track["track"]["artists"][0]["name"])
        song_features["artist_pop"].append(artist_info["popularity"])
        song_features["artist_genres"].append(artist_info["genres"])
        
        #Album
        song_features["album"].append(track["track"]["album"]["name"])
        
        #Popularity of the track
        song_features["track_pop"].append(track["track"]["popularity"])
        #song_features.update(sp.audio_features(track_uri)[0])
        for k,v in sp.audio_features(track_uri)[0].items():
            #print(type(k),v)
            song_features[k].append(v)
        #print(song_features['id'])
        #print(len(song_features["artist_info"]))
    return song_features

d=get_playlist()
df_song_features = pd.DataFrame(d)
#print(df_song_features.head(5))


def drop_duplicates(df):
    '''
    Drop duplicate songs
    '''
    df['artists_song'] = df.apply(lambda row: row['artist_name']+row['track_name'],axis = 1)
    return df.drop_duplicates('artists_song')

    

def select_cols(df):
    '''
    Select useful columns
    '''
    return df[['artist_name','id','track_name','danceability', 'energy', 'key', 'loudness', 'mode',
    'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', "artist_pop", "artist_genres", "track_pop"]]

songDF = drop_duplicates(df_song_features)
songDF = select_cols(songDF)
songDF.head()
#print(songDF.artist_genres.to_string(index=False))

def ohe_prep(df, column, new_name): 
    ''' 
    Create One Hot Encoded features of a specific column
    ---
    Input: 
    df (pandas dataframe): Spotify Dataframe
    column (str): Column to be processed
    new_name (str): new column name to be used
        
    Output: 
    tf_df: One-hot encoded features 
    '''
    
    tf_df = pd.get_dummies(df[column])
    feature_names = tf_df.columns
    tf_df.columns = [new_name + "|" + str(i) for i in feature_names]
    tf_df.reset_index(drop = True, inplace = True)    
    return tf_df

from textblob import TextBlob

def getSubjectivity(text):
  '''
  Getting the Subjectivity using TextBlob
  '''
  return TextBlob(text).sentiment.subjectivity

def getPolarity(text):
  '''
  Getting the Polarity using TextBlob
  '''
  return TextBlob(text).sentiment.polarity

def getAnalysis(score, task="polarity"):
  '''
  Categorizing the Polarity & Subjectivity score
  '''
  if task == "subjectivity":
    if score < 1/3:
      return "low"
    elif score > 1/3:
      return "high"
    else:
      return "medium"
  else:
    if score < 0:
      return 'Negative'
    elif score == 0:
      return 'Neutral'
    else:
      return 'Positive'

def sentiment_analysis(df, text_col):
  '''
  Perform sentiment analysis on text
  ---
  Input:
  df (pandas dataframe): Dataframe of interest
  text_col (str): column of interest
  '''
  df['subjectivity'] = df[text_col].apply(getSubjectivity).apply(lambda x: getAnalysis(x,"subjectivity"))
  df['polarity'] = df[text_col].apply(getPolarity).apply(getAnalysis)
  return df


def create_feature_set(df, float_cols):
    '''
    Process spotify df to create a final set of features that will be used to generate recommendations
    ---
    Input: 
    df (pandas dataframe): Spotify Dataframe
    float_cols (list(str)): List of float columns that will be scaled
            
    Output: 
    final (pandas dataframe): Final set of features 
    '''
    
    # Tfidf genre lists
    tfidf = TfidfVectorizer()
    tfidf_matrix =  tfidf.fit_transform(df['artist_genres'].apply(lambda x: " ".join(x)))
    genre_df = pd.DataFrame(tfidf_matrix.toarray())
    genre_df.columns = ['genre' + "|" + i for i in tfidf.get_feature_names()]
    if 'genre|unknown' in genre_df:
        genre_df.drop(columns='genre|unknown') # drop unknown genre
        genre_df.reset_index(drop = True, inplace=True)
    
    # Sentiment analysis
    df = sentiment_analysis(df, "track_name")

    # One-hot Encoding
    subject_ohe = ohe_prep(df, 'subjectivity','subject') * 0.3
    polar_ohe = ohe_prep(df, 'polarity','polar') * 0.5
    key_ohe = ohe_prep(df, 'key','key') * 0.5
    mode_ohe = ohe_prep(df, 'mode','mode') * 0.5

    # Normalization
    # Scale popularity columns
    pop = df[["artist_pop","track_pop"]].reset_index(drop = True)
    scaler = MinMaxScaler()
    pop_scaled = pd.DataFrame(scaler.fit_transform(pop), columns = pop.columns) * 0.2 

    # Scale audio columns
    floats = df[float_cols].reset_index(drop = True)
    scaler = MinMaxScaler()
    floats_scaled = pd.DataFrame(scaler.fit_transform(floats), columns = floats.columns) * 0.2

    # Concanenate all features
    final = pd.concat([genre_df, floats_scaled, pop_scaled, subject_ohe, polar_ohe, key_ohe, mode_ohe], axis = 1)
    
    # Add song id
    final['id']=df['id'].values
    
    return final

def generate_playlist_feature(complete_feature_set, playlist_df):
    '''
    Summarize a user's playlist into a single vector
    ---
    Input: 
    complete_feature_set (pandas dataframe): Dataframe which includes all of the features for the spotify songs
    playlist_df (pandas dataframe): playlist dataframe
        
    Output: 
    complete_feature_set_playlist_final (pandas series): single vector feature that summarizes the playlist
    complete_feature_set_nonplaylist (pandas dataframe): 
    '''
    
    # Find song features in the playlist
    complete_feature_set_playlist = complete_feature_set[complete_feature_set['id'].isin(playlist_df['id'].values)]
    # Find all non-playlist song features
    complete_feature_set_nonplaylist = complete_feature_set[~complete_feature_set['id'].isin(playlist_df['id'].values)]
    complete_feature_set_playlist_final = complete_feature_set_playlist.drop(columns = "id")
    return complete_feature_set_playlist_final.sum(axis = 0), complete_feature_set_nonplaylist


def generate_playlist_recos(df, features, nonplaylist_features):
    '''
    Generated recommendation based on songs in aspecific playlist.
    ---
    Input: 
    df (pandas dataframe): spotify dataframe
    features (pandas series): summarized playlist feature (single vector)
    nonplaylist_features (pandas dataframe): feature set of songs that are not in the selected playlist
        
    Output: 
    non_playlist_df_top_40: Top 40 recommendations for that playlist
    '''
    
    non_playlist_df = df[df['id'].isin(nonplaylist_features['id'].values)]
    # Find cosine similarity between the playlist and the complete song set
    non_playlist_df['sim'] = cosine_similarity(nonplaylist_features.drop('id', axis = 1).values, features.values.reshape(1, -1))[:,0]
    non_playlist_df_top_40 = non_playlist_df.sort_values('sim',ascending = False).head(40)
    
    return non_playlist_df_top_40

playlist_df = pd.DataFrame(d['id'])
float_columns = list(sp.audio_features(d['track_uri'][0])[0].keys())
finaldf = create_feature_set(songDF, float_columns)


playlist_features, nonplaylist_features = generate_playlist_feature(finaldf, playlist_df)

recs = generate_playlist_recos(finaldf, playlist_features, nonplaylist_features)

recs.head()
