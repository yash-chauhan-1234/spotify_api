import pandas as pd
from collections import defaultdict
import numpy as np
from functools import reduce


df_main=None



def create_df_liked(sp):
    name=[]
    id=[]
    duration=[]
    popularity=[]
    album=[]
    artist=[]

    
    for j in range(5):
        for i in range(50):
            results=sp.current_user_saved_tracks(limit=50, offset=50*j)["items"][i]["track"]
            name.append(results["name"])
            id.append(results["id"])
            duration.append(results["duration_ms"])
            popularity.append(results["popularity"])
            album.append(results["album"]["name"])
            artist.append(results["artists"][0]["name"])


    df=pd.DataFrame({"name": name, "id":id, "duration":duration, "popularity":popularity, "album":album, "artist":artist})



    return df


def get_audio_features2(songs_df, sp):

    audio_features=pd.DataFrame(columns=["name", "id", "danceability", "energy", "key", "loudness", "mode", "speechiness", "acousticness", "instrumentalness", "liveness", "valence",	"tempo"])
    for i in range(5):
        for j in range(50):
            df_temp=songs_df.loc[i*j:(i+1)*j, ["name", "id"]]
    
            features=pd.DataFrame.from_dict(sp.audio_features(df_temp["id"]))[["id", "danceability", "energy", "key", "loudness", "mode", "speechiness", "acousticness", "instrumentalness", "liveness", "valence",	"tempo"]]
        temp=pd.merge(df_temp, features, on="id", how="inner")
        audio_features=pd.concat([audio_features, temp], ignore_index=True)

    audio_features=audio_features.astype({"danceability":"float64", "energy":"float64", "key":"int64", "loudness":"float64", "mode":"int64", "speechiness":"float64", "acousticness":"float64", "instrumentalness":"float64", "liveness":"float64", "valence":"float64", "tempo":"float64"})

    return audio_features
    
def get_audio_features(songs_df, sp):
    df_temp=songs_df[["name", "id"]]
    features=pd.DataFrame.from_dict(sp.audio_features(songs_df["id"]))[["danceability", "energy", "key", "loudness", "mode", "speechiness", "acousticness", "instrumentalness", "liveness", "valence",	"tempo"]]
    audio_features=pd.concat([df_temp, features], axis=1)

    return audio_features
    

def get_playlist(sp):
    playlists=sp.featured_playlists()["playlists"]
    df1=pd.DataFrame.from_dict(playlists["items"])[["id", "name"]]

    playlist=defaultdict(list)
    for item in playlists["items"]:
        playlist["count"].append(item["tracks"]["total"])
    df2=pd.DataFrame.from_dict(playlist)
    
    #df2
    playlists_rec=pd.concat([df1, df2], axis=1)
    return playlists_rec
    
    

def playlist_tracks(playlist_id, sp):
    tracks=sp.playlist_tracks(playlist_id)["items"]
    track_df=defaultdict(list)
    for track in tracks:
        track_df["id"].append(track["track"]["id"])
        track_df["name"].append(track["track"]["name"])
    return pd.DataFrame.from_dict(track_df)


def audio_features_playlist_mean(playlist_ids, sp):
    all_playlist=[]
    for playlist_id in playlist_ids:
        tracks=playlist_tracks(playlist_id, sp)
        all_playlist.append(pd.DataFrame(get_audio_features(tracks,sp).mean(), columns=[playlist_id]))

    return all_playlist



def x_y(dataframes, df_fav):
    X = reduce(lambda left,right: pd.merge(left,right, left_index=True, right_index=True), dataframes)
    X.drop(["mode"], inplace=True, axis=0)
    
    Y = pd.DataFrame(df_fav.median(), columns= ['fav_playlist'])
    Y= Y.drop('mode')

    return X, Y


#ML STARTS HERE

def randomforest(X, Y):
    # Analyze feature importances
    from sklearn.ensemble import RandomForestRegressor
    import numpy as np
    # Can combine step above with this
    forest = RandomForestRegressor(max_depth=7, max_features=0.5) 
    forest.fit(X,Y)
    importances1 = forest.feature_importances_
    indices1 = np.argsort(importances1)[::-1]
    # Print the feature rankings
    # print("Playlist ranking:")
    
    # for f in range(len(importances1)):
    #     print("%d. %s %f " % (f + 1, 
    #             X.columns[indices1[f]], 
    #             importances1[indices1[f]]))

    rfr=pd.DataFrame({"no": [f+1 for f in range(len(importances1))],
                    "name":[X.columns[indices1[f]] for f in range(len(importances1))],
                    "importance": [importances1[indices1[f]] for f in range(len(importances1))]})
    return rfr

def adaboost(X, Y):
    from sklearn.ensemble import AdaBoostRegressor
    import numpy as np

    adaboost=AdaBoostRegressor(n_estimators=100)
    adaboost.fit(X=X,y=Y)

    importances2 = adaboost.feature_importances_
    indices2 = np.argsort(importances2)[::-1]
    # Print the feature rankings
    # print("Playlist ranking:")
    
    # for f in range(len(importances2)):
    #     print("%d. %s %f " % (f + 1, 
    #             X.columns[indices2[f]], 
    #             importances2[indices2[f]]))

    abr=pd.DataFrame({"no": [f+1 for f in range(len(importances2))],
                    "name":[X.columns[indices2[f]] for f in range(len(importances2))],
                    "importance": [importances2[indices2[f]] for f in range(len(importances2))]})
    return abr

def gradientboost(X, Y):
    from sklearn.ensemble import GradientBoostingRegressor
    import numpy as np

    gbr=GradientBoostingRegressor(n_estimators=100, learning_rate=0.18, max_depth=7, max_features=1.0)
    gbr.fit(X, Y)

    importances3 = gbr.feature_importances_
    indices3 = np.argsort(importances3)[::-1]
    # Print the feature rankings
    print("Playlist ranking:")
    
    # for f in range(len(importances3)):
    #     print("%d. %s %f " % (f + 1, 
    #             X.columns[indices3[f]], 
    #             importances3[indices3[f]]))

    gbr=pd.DataFrame({"no": [f+1 for f in range(len(importances3))],
                    "name":[X.columns[indices3[f]] for f in range(len(importances3))],
                    "importance": [importances3[indices3[f]] for f in range(len(importances3))]})
    return gbr

def xgb(X, Y):
    import xgboost as xgb
    import matplotlib.pyplot as plt

    dmatrix=xgb.DMatrix(data=X, label=Y)
    params_linear = {"booster":"gblinear"} 
    xgb_reg_linear=xgb.train(params=params_linear, dtrain=dmatrix, num_boost_round=10)
    params_tree={"objective":"reg:linear", "colsamply_bytree":0.5, 'max_depth': 9}
    xgb_reg_tree=xgb.train(params=params_tree, dtrain=dmatrix, num_boost_round=10)

    fig1, ax1=plt.subplots()
    ax1=xgb.plot_importance(xgb_reg_linear)
    plt.rcParams["figure.figsize"]=[5,5]
    
    fig2, ax2=plt.subplots()
    ax2=xgb.plot_importance(xgb_reg_tree)
    plt.rcParams["figure.figsize"]=[5,5]

    return fig1, fig2

def final(rfr, abr, gbr):
    df35=rfr.merge(abr.merge(gbr, on="name"), on="name")
    df35["no_final"]=df35["no"]+df35["no_x"]+df35["no_y"]
    df35.sort_values(by="no_final")
    return df35["name"]

from sklearn.cluster import KMeans
class recommend2:

    def __init__(self, df_fav, sp):
        self.flag=False
        self.df_fav=df_fav
        self.sp=sp
        self.kmeans=None
        self.cluster_map = pd.DataFrame()


    def kmeans_algo(self):
        x2=self.df_fav.drop(["name", "id", "mode", "speechiness", "instrumentalness", "liveness"], axis=1)
        print(x2.columns)
        self.kmeans=KMeans(n_clusters=8).fit(x2)

        
        self.cluster_map['name'] = self.df_fav["name"]
        self.cluster_map['cluster'] = self.kmeans.labels_
        # return self.kmeans, self.cluster_map


    def prediction(self, name, artist):
        track_id=self.sp.search(q='artist:' + artist + ' track:' + name, type='track')["tracks"]["items"][0]["id"]
        features=np.delete(np.array(list(self.sp.audio_features(track_id)[0].values())[:11]), [4,5,7,8]).reshape(1,-1)
    
        preds=self.kmeans.predict(features)

        # return self.cluster_map, preds
        return self.cluster_map.loc[self.cluster_map.cluster==int(preds)]["name"]
    # return preds
    

    
    

    
