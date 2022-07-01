import streamlit as st
# import pandas as pd
# import numpy as np
from apps import playlist, rec, viz
from multiapp import MultiApp
st.set_page_config(layout="wide")

st.title("Spotify")

with st.expander("Instructions:"):
    st.write("Perform the following steps: \n1. Create a spotify developer account at https://developer.spotify.com/dashboard/login \n2. Click on Create an App. \n3. Give it a name and/or a description. \n4. You'll get the Client ID and the Client Secret ID in this dashboard. \n5. Get the username from the spotify app itself.")


app=MultiApp()

app.add_app("Playlist", playlist.app)
app.add_app("Recommend", rec.app)
app.add_app("Visualizations", viz.app)

app.run()