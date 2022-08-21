import spotipy
import spotipy.util
from spotipy.oauth2 import SpotifyClientCredentials
import requests
import base64


#Authentication - without user
def authenticate():
    with open(r"C:\Users\kayys\PracticalDataScience-ENCA\secret.txt", 'r') as f:
        secret_ls = f.readlines()
        cid = secret_ls[0][:-1]
        secret = secret_ls[1]
        username = secret_ls[2]
        f.close()


    cid=cid.strip()
    secret=secret.strip()

    scope='user-library-read'

    token = spotipy.util.prompt_for_user_token(username,
                            scope,
                            client_id=cid,
                            client_secret=secret,
                            redirect_uri='http://localhost:8888/callback')

    return(spotipy.Spotify(auth=token))

sp=authenticate()
