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
top_five_artists = df['artist'].value_counts().head(5)
st.bar_chart(top_five_artists)

# Top 5 Loudest Tracks
st.subheader("Top 5 Loudest Tracks")
top_loudest_tracks = df.nlargest(5, 'loudness')[['song_title', 'artist', 'loudness']]
st.write(top_loudest_tracks)

# Artist with Most Danceable Song
st.subheader("Artist with Most Danceable Song")
most_danceable_song = df.loc[df['danceability'].idxmax()]
st.write(f"Artist: **{most_danceable_song['artist']}**")
st.write(f"Song Title: **{most_danceable_song['song_title']}**")
st.write(f"Danceability: **{most_danceable_song['danceability']}**")

# Top 10 Instrumental Tracks
st.subheader("Top 10 Instrumental Tracks")
top_instrumental_tracks = df.nlargest(10, 'instrumentalness')[['song_title', 'artist', 'instrumentalness']]
st.write(top_instrumental_tracks)

# Feature Distributions
st.subheader("Feature Distributions")
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
sns.histplot(df['energy'], bins=30, ax=axs[0, 0], color='blue')
axs[0, 0].set_title('Energy')
sns.histplot(df['valence'], bins=30, ax=axs[0, 1], color='red')
axs[0, 1].set_title('Valence')
sns.histplot(df['tempo'], bins=30, ax=axs[1, 0], color='green')
axs[1, 0].set_title('Tempo')
sns.histplot(df['loudness'], bins=30, ax=axs[1, 1], color='purple')
axs[1, 1].set_title('Loudness')
st.pyplot(fig)

# Top 10 Energetic Tracks
st.subheader("Top 10 Energetic Tracks")
top_energetic_tracks = df.nlargest(10, 'energy')[['song_title', 'artist', 'energy']]
st.write(top_energetic_tracks)

# Top 10 Tracks with Most Valence
st.subheader("Top 10 Tracks with Most Valence")
top_valence_tracks = df.nlargest(10, 'valence')[['song_title', 'artist', 'valence']]
st.write(top_valence_tracks)

# Most Common Track Duration
st.subheader("Most Common Track Duration")
most_common_duration = df['duration_ms'].mode()[0]
minutes = most_common_duration // 60000
seconds = (most_common_duration % 60000) // 1000
st.write(f"The most common track duration is **{minutes} minutes and {seconds} seconds** ({most_common_duration} ms).")

# Most Trending Artist (Proxy for Genre)
st.subheader("Most Trending Artist")
most_trending_artist = df['artist'].value_counts().idxmax()
most_trending_artist_count = df['artist'].value_counts().max()
st.write(f"The most trending artist is **{most_trending_artist}** with **{most_trending_artist_count}** tracks in the dataset.")

# Correlation Heatmap
st.subheader("Correlation Heatmap (Numeric Features)")
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
corr = df[numeric_cols].corr()
fig2, ax2 = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="crest", ax=ax2)
st.pyplot(fig2)

# Energy vs. Danceability
st.subheader("Energy vs. Danceability")
fig3, ax3 = plt.subplots()
sns.scatterplot(data=df, x='energy', y='danceability', alpha=0.3, ax=ax3)
st.pyplot(fig3)

# Additional Feature Distributions
st.subheader("Additional Feature Distributions")
fig4, axs4 = plt.subplots(2, 2, figsize=(12, 10))
sns.histplot(df['acousticness'], bins=30, ax=axs4[0, 0], color='cyan')
axs4[0, 0].set_title('Acousticness')
sns.histplot(df['instrumentalness'], bins=30, ax=axs4[0, 1], color='magenta')
axs4[0, 1].set_title('Instrumentalness')
sns.histplot(df['liveness'], bins=30, ax=axs4[1, 0], color='yellow')
axs4[1, 0].set_title('Liveness')
sns.histplot(df['speechiness'], bins=30, ax=axs4[1, 1], color='orange')
axs4[1, 1].set_title('Speechiness')
st.pyplot(fig4)

# Top 10 Tracks by Liveness
st.subheader("Top 10 Tracks by Liveness")
top_liveness_tracks = df.nlargest(10, 'liveness')[['song_title', 'artist', 'liveness']]
st.write(top_liveness_tracks)

# Top 10 Tracks by Acousticness
st.subheader("Top 10 Tracks by Acousticness")
top_acousticness_tracks = df.nlargest(10, 'acousticness')[['song_title', 'artist', 'acousticness']]
st.write(top_acousticness_tracks)

# Top 10 Tracks by Speechiness
st.subheader("Top 10 Tracks by Speechiness")
top_speechiness_tracks = df.nlargest(10, 'speechiness')[['song_title', 'artist', 'speechiness']]
st.write(top_speechiness_tracks)

st.info("This dashboard covers all major Spotify EDA insights. You can further extend it with filters, more plots, or interactive widgets as needed!")
