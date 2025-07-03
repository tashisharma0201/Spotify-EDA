import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Spotify EDA Dashboard", layout="wide")
st.title("Spotify Exploratory Data Analysis")

@st.cache_data
def load_data():
    df = pd.read_csv("data.csv")
    df.columns = df.columns.str.strip()
    return df

df = load_data()

st.header("Dataset Preview")
st.dataframe(df.head())

# Top 5 Popular Artists
st.subheader("Top 5 Popular Artists")
top_five_artists = df.groupby("artist").count().sort_values(by="song_title", ascending=False)["song_title"][:5]
st.bar_chart(top_five_artists)

# Top 5 Loudest Tracks
st.subheader("Top 5 Loudest Tracks")
top_loudest_tracks = df.nlargest(5, 'loudness')[['song_title', 'loudness']]
st.write(top_loudest_tracks)

# Artist with Most Danceable Song
st.subheader("Artist with Most Danceable Song")
most_danceable_song = df.loc[df['danceability'].idxmax()]
st.write(f"Artist: {most_danceable_song['artist']}")
st.write(f"Song Title: {most_danceable_song['song_title']}")
st.write(f"Danceability: {most_danceable_song['danceability']}")

# Top 10 Instrumental Tracks
st.subheader("Top 10 Instrumental Tracks")
top_instrumental_tracks = df.nlargest(10, 'instrumentalness')[['song_title', 'instrumentalness']]
st.write(top_instrumental_tracks)

# Feature Distributions
st.subheader("Feature Distributions")
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
sns.histplot(df['energy'], bins=30, ax=axs[0, 0], color='blue')
axs[0, 0].set_title('Energy Distribution')
sns.histplot(df['valence'], bins=30, ax=axs[0, 1], color='red')
axs[0, 1].set_title('Valence Distribution')
sns.histplot(df['tempo'], bins=30, ax=axs[1, 0], color='green')
axs[1, 0].set_title('Tempo Distribution')
sns.histplot(df['loudness'], bins=30, ax=axs[1, 1], color='purple')
axs[1, 1].set_title('Loudness Distribution')
st.pyplot(fig)

# Top 10 Energetic Tracks
st.subheader("Top 10 Energetic Tracks")
top_energetic_tracks = df.nlargest(10, 'energy')[['song_title', 'energy']]
st.write(top_energetic_tracks)

# Top 10 Tracks with Most Valence
st.subheader("Top 10 Tracks with Most Valence")
top_valence_tracks = df.nlargest(10, 'valence')[['song_title', 'valence']]
st.write(top_valence_tracks)

# Most Common Track Duration
st.subheader("Most Common Track Duration")
most_common_duration = df['duration_ms'].mode()[0]
minutes = most_common_duration // 60000
seconds = (most_common_duration % 60000) // 1000
st.write(f"The most common track duration is **{minutes} minutes and {seconds} seconds** ({most_common_duration} ms).")

# Most Trending Artist (as proxy for Genre)
st.subheader("Most Trending Artist")
most_trending_artist = df['artist'].value_counts().idxmax()
most_trending_artist_count = df['artist'].value_counts().max()
st.write(f"The most trending artist is **{most_trending_artist}** with **{most_trending_artist_count}** tracks in the dataset.")

st.info("Explore more insights by extending this dashboard with additional plots and filters!")
