import streamlit as st
import requests
import pandas as pd

API_URL = "http://127.0.0.1:8000/recommend"

# -------------------------
# UI setup
# -------------------------
st.set_page_config(page_title="Spotify Song Recommender", page_icon="🎵")

st.markdown(
    "<h1 style='text-align:center;'>🎵 Spotify Song Recommender</h1>",
    unsafe_allow_html=True
)
st.caption("Discover songs that sound similar")

# -------------------------
# Load data (for dropdown only)
# -------------------------
df = pd.read_csv("spotify.csv")

track = st.selectbox("🎧 Choose a song you like", df["track_name"].unique())
top_n = st.slider("Number of recommendations", 3, 10, 5)

# -------------------------
# Call API
# -------------------------
if st.button("🔍 Recommend Songs"):
    with st.spinner("Finding similar songs..."):
        response = requests.post(
            API_URL,
            json={"track_name": track, "top_n": top_n},
            timeout=5
        )

        if response.status_code == 200:
            data = response.json()["recommendations"]

            st.success("🎶 You might also like:")

            for song in data:
                st.markdown(
                    f"""
                    **{song['track_name']}**  
                    *{song['artist']}*  
                    Genre: `{song['genre']}` • Playlist: `{song['playlist_category']}`
                    ---
                    """
                )
        else:
            st.error(response.text)
