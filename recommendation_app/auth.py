import spotipy
import spotipy.util
from spotipy.oauth2 import SpotifyClientCredentials
import requests
import base64
import pandas as pd

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
    scope='playlist-modify-public'


    #cid = str(base64.b64encode(bytes(cid, 'utf-8')))
    #secret = str(base64.b64encode(bytes(secret, 'utf-8')))

    """token = spotipy.util.prompt_for_user_token(username,
                            scope,
                            client_id='cid',
                            client_secret='secret',
                            redirect_uri='http://localhost:8888/callback')

    sp=spotipy.Spotify(auth=token)"""
    #if token:
    client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
    client_credentials_manager = SpotifyClientCredentials(client_id='64fd2a7e6edd467cbe8f2a8e099a4eab', client_secret='b4246dc052dd4fcda242b37c08e3ec07')
    sp = spotipy.Spotify(client_credentials_manager = client_credentials_manager)


    playlist_link = "https://open.spotify.com/playlist/37i9dQZEVXbNG2KDcFcKOF?si=1333723a6eff4b7f"
    playlist_URI = playlist_link.split("/")[-1].split("?")[0]
    track_uris = [x["track"]["uri"] for x in sp.playlist_tracks(playlist_URI)["items"]]
    column_names = ['artist_name','id','track_name','danceability', 'energy', 'key', 'loudness', 'mode',
    'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', "artist_pop", "genres", "track_pop"]
    df_features = pd.DataFrame()
    

    playlist_length= len(sp.playlist_tracks(playlist_URI)["items"])
    for track in range(0, playlist_length):
        pl = sp.playlist_tracks(playlist_URI)["items"]
        #URI
        dfl= df_features.loc[track]
        dfl['track_uri'] = track_uri =  pl[track]["track"]["uri"]
        
        #Track name
        df_features.loc[track]["track_name"] = pl[track]["track"]["name"]
        
        #Main Artist
        df_features.loc[track]["artist_uri"] = artist_uri = pl[track]["track"]["artists"][0]["uri"]
        #df_features["artist_info"] = artist_info = sp.artist(artist_uri)
        
        #Name, popularity, genre
        df_features.loc[track]["artist_name"] = pl[track]["track"]["artists"][0]["name"]
        #df_features["artist_pop"] = artist_info["popularity"]
        #df_features["artist_genres"] = 
        #print(artist_info["genres"])
        
        #Album
        df_features.loc[track]["album"] = pl[track]["track"]["album"]["name"]
        
        #Popularity of the track
        df_features.loc[track]["track_pop"] = pl[track]["track"]["popularity"]
        df = pd.DataFrame.from_dict([sp.audio_features(track_uri)[0]])

        df_features.loc[track] = pd.concat([df_features.loc[track], df])

    print(df_features)

    """songDF = drop_duplicates(df_features)
    songDF = select_cols(songDF)
    songDF.head()
"""

"""
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
    'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', "artist_pop", "genres", "track_pop"]]
    """