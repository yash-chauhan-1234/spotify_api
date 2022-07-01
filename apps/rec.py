import streamlit as st
import pandas as pd
import script

def authorize():
    import spotipy
    import spotipy.oauth2 as oauth2
    import spotipy.util as util

    

    scope="user-library-read"
    redirect_uri="http://localhost:8080"
    CLIENT_ID=st.text_input(label="Enter your client id", value="442994a7d993428b878da8753d958dd5")
    CLIENT_SECRET=st.text_input(label="Enter your secret id", value="974319b88aae429883f9074a4a748cdb")
    username=st.text_input(label="Enter your username", value="1q2mr4wzdxbyfec3fixti7hvc")
    sp=None
    # if st.button("Submit"):
    if CLIENT_ID and CLIENT_SECRET and username:
        token=util.prompt_for_user_token(
                username=username,
                scope=scope,
                client_id=CLIENT_ID,
                client_secret=CLIENT_SECRET,
                redirect_uri=redirect_uri
            )
        sp=spotipy.Spotify(auth=token)
        return sp

@st.experimental_singleton
def df_maker(_sp):
    df=script.create_df_liked(_sp)

    df_fav=script.get_audio_features2(df, _sp)

    return df_fav


def app():
    sp=authorize()

    if sp:
        df_fav=df_maker(sp)
        # st.dataframe(df_fav)
        script.df_main=df_fav
        
        with st.form(key="myForm"):
            name=st.text_input(label="Enter Name of the song", key="name")
            artist=st.text_input(label="Enter artist of the song", key="artist")
            submit=st.form_submit_button(label="Submit")

            if submit:
                st.markdown("---")
                
                st.write("The following songs in your playlist are just like "+name+" by "+artist)
                r=script.recommend2(df_fav, sp)
                r.kmeans_algo()
                st.dataframe(r.prediction(name, artist))
