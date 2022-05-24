import streamlit as st
import script
import seaborn as sns
import matplotlib.pyplot as plt

@st.cache
def get_data(sp):
    df=script.create_df_liked(sp)

        
    df_fav=script.get_audio_features2(df, sp)

    return df_fav

def app():
    import spotipy
    import spotipy.oauth2 as oauth2
    import spotipy.util as util

    

    scope="user-library-read"
    redirect_uri="http://localhost:8080"
    CLIENT_ID=st.text_input(label="Enter your client id", value="529b46cf46d84dec9969f39e75bdf07f")
    CLIENT_SECRET=st.text_input(label="Enter your secret id", value="8013d63c16b64ff489e63fe677726cc2")
    username=st.text_input(label="Enter your username", value="1q2mr4wzdxbyfec3fixti7hvc")
    sp=None
    if st.button("Submit"):
        token=util.prompt_for_user_token(
                username=username,
                scope=scope,
                client_id=CLIENT_ID,
                client_secret=CLIENT_SECRET,
                redirect_uri=redirect_uri
            )
        sp=spotipy.Spotify(auth=token)


    # app.add_app("login", login.app)
    # sp=login.app()
    # app.add_app("data", data.app(sp))

    # app.run()

        st.markdown("---")

        df_fav=get_data(sp)        
        script.df_main=df_fav




        playlist_fav=script.get_playlist(sp)

        dataframes=script.audio_features_playlist_mean(playlist_fav["id"], sp)

        X, Y = script.x_y(dataframes, df_fav)

        rfr=script.randomforest(X, Y)
        abr=script.adaboost(X, Y)
        gbr=script.gradientboost(X, Y)
        # lin, tre=script.xgb(X,Y)

        # with st.expander("See XGBoost ranking:"):
        #     c1, c2=st.columns(2)
            
        #     c1.pyplot(lin, caption="Linear XGBoost")
        #     c2.pyplot(tre, caption="Tree XGBoost")
        

        final_list=script.final(rfr, abr, gbr)
        # final_list

        st.write("The playlist we curated is as follows:")
        st.dataframe(script.playlist_tracks(final_list[0], sp)["name"])
