import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Spotify EDA Dashboard", layout="wide")

# --- Custom CSS for background and headers ---
st.markdown("""
    <style>
    .stApp {background-color: #f5f7fa;}
    h1, h2, h3, h4 {color: #22223b; font-weight: bold;}
    .css-1d391kg {color: #22223b;}
    </style>
""", unsafe_allow_html=True)

st.title("🎵 Spotify Exploratory Data Analysis Dashboard")

@st.cache_data
def load_data():
    df = pd.read_csv("data.csv")
    df.columns = df.columns.str.strip()
    return df

df = load_data()

# --- Sidebar filter for artist selection and multi-feature selection ---
st.sidebar.title("🔎 Filter Options")
artist_options = ['All'] + sorted(df['artist'].unique())
selected_artist = st.sidebar.selectbox("🎤 Select Artist", artist_options)
df_filtered = df if selected_artist == 'All' else df[df['artist'] == selected_artist]

# Multi-feature selection for overlayed distributions
feature_cols = ['acousticness', 'danceability', 'duration_ms', 'energy', 'instrumentalness',
                'liveness', 'loudness', 'speechiness', 'tempo', 'valence']
selected_features = st.sidebar.multiselect(
    "📊 Select features to compare distributions",
    feature_cols,
    default=['energy', 'danceability']
)

st.header("🗂️ Dataset Preview")
st.dataframe(df_filtered.head())

# Top 5 Popular Artists
st.subheader("🏆 Top 5 Popular Artists")
top_five_artists = df_filtered['artist'].value_counts().head(5)
st.bar_chart(top_five_artists)

# Top 5 Loudest Tracks
st.subheader("🔊 Top 5 Loudest Tracks")
top_loudest_tracks = df_filtered.nlargest(5, 'loudness')[['song_title', 'artist', 'loudness']]
st.write(top_loudest_tracks)

# Artist with Most Danceable Song
st.subheader("💃 Artist with Most Danceable Song")
if not df_filtered.empty:
    most_danceable_song = df_filtered.loc[df_filtered['danceability'].idxmax()]
    st.write(f"Artist: **{most_danceable_song['artist']}**")
    st.write(f"Song Title: **{most_danceable_song['song_title']}**")
    st.write(f"Danceability: **{most_danceable_song['danceability']}**")
else:
    st.write("No data available.")

# Top 10 Instrumental Tracks
st.subheader("🥁 Top 10 Instrumental Tracks")
top_instrumental_tracks = df_filtered.nlargest(10, 'instrumentalness')[['song_title', 'artist', 'instrumentalness']]
st.write(top_instrumental_tracks)

# --- Multi-feature overlayed histograms ---
if selected_features:
    st.subheader("📊 Compare Multiple Feature Distributions")
    fig, ax = plt.subplots(figsize=(10, 6))
    for feature in selected_features:
        sns.histplot(df_filtered[feature], bins=30, kde=True, ax=ax, label=feature, alpha=0.5)
    ax.set_title("Overlayed Distributions")
    ax.legend()
    st.pyplot(fig)

# Feature Distributions (as in notebook)
st.subheader("📊 Feature Distributions (Energy, Valence, Tempo, Loudness)")
fig2, axs = plt.subplots(2, 2, figsize=(12, 10))
sns.histplot(df_filtered['energy'], bins=30, ax=axs[0, 0], color='blue')
axs[0, 0].set_title('Energy')
sns.histplot(df_filtered['valence'], bins=30, ax=axs[0, 1], color='red')
axs[0, 1].set_title('Valence')
sns.histplot(df_filtered['tempo'], bins=30, ax=axs[1, 0], color='green')
axs[1, 0].set_title('Tempo')
sns.histplot(df_filtered['loudness'], bins=30, ax=axs[1, 1], color='purple')
axs[1, 1].set_title('Loudness')
st.pyplot(fig2)

# Top 10 Energetic Tracks
st.subheader("⚡ Top 10 Energetic Tracks")
top_energetic_tracks = df_filtered.nlargest(10, 'energy')[['song_title', 'artist', 'energy']]
st.write(top_energetic_tracks)

# Top 10 Tracks with Most Valence
st.subheader("😊 Top 10 Tracks with Most Valence")
top_valence_tracks = df_filtered.nlargest(10, 'valence')[['song_title', 'artist', 'valence']]
st.write(top_valence_tracks)

# Most Common Track Duration
st.subheader("⏳ Most Common Track Duration")
if not df_filtered.empty:
    most_common_duration = df_filtered['duration_ms'].mode()[0]
    minutes = most_common_duration // 60000
    seconds = (most_common_duration % 60000) // 1000
    st.write(f"The most common track duration is **{minutes} minutes and {seconds} seconds** ({most_common_duration} ms).")
else:
    st.write("No data available.")

# Most Trending Artist (Proxy for Genre)
st.subheader("🔥 Most Trending Artist")
if not df_filtered.empty:
    most_trending_artist = df_filtered['artist'].value_counts().idxmax()
    most_trending_artist_count = df_filtered['artist'].value_counts().max()
    st.write(f"The most trending artist is **{most_trending_artist}** with **{most_trending_artist_count}** tracks in the dataset.")
else:
    st.write("No data available.")

# Correlation Heatmap
st.subheader("🧮 Correlation Heatmap (Numeric Features)")
numeric_cols = df_filtered.select_dtypes(include=['float64', 'int64']).columns
corr = df_filtered[numeric_cols].corr()
fig3, ax3 = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="crest", ax=ax3)
st.pyplot(fig3)

# Energy vs. Danceability
st.subheader("⚡ Energy vs. Danceability")
fig4, ax4 = plt.subplots()
sns.scatterplot(data=df_filtered, x='energy', y='danceability', alpha=0.3, ax=ax4)
st.pyplot(fig4)

# Additional Feature Distributions
st.subheader("📊 Additional Feature Distributions")
fig5, axs5 = plt.subplots(2, 2, figsize=(12, 10))
sns.histplot(df_filtered['acousticness'], bins=30, ax=axs5[0, 0], color='cyan')
axs5[0, 0].set_title('Acousticness')
sns.histplot(df_filtered['instrumentalness'], bins=30, ax=axs5[0, 1], color='magenta')
axs5[0, 1].set_title('Instrumentalness')
sns.histplot(df_filtered['liveness'], bins=30, ax=axs5[1, 0], color='yellow')
axs5[1, 0].set_title('Liveness')
sns.histplot(df_filtered['speechiness'], bins=30, ax=axs5[1, 1], color='orange')
axs5[1, 1].set_title('Speechiness')
st.pyplot(fig5)

# Top 10 Tracks by Liveness
st.subheader("🎤 Top 10 Tracks by Liveness")
top_liveness_tracks = df_filtered.nlargest(10, 'liveness')[['song_title', 'artist', 'liveness']]
st.write(top_liveness_tracks)

# Top 10 Tracks by Acousticness
st.subheader("🎸 Top 10 Tracks by Acousticness")
top_acousticness_tracks = df_filtered.nlargest(10, 'acousticness')[['song_title', 'artist', 'acousticness']]
st.write(top_acousticness_tracks)

# Top 10 Tracks by Speechiness
st.subheader("🗣️ Top 10 Tracks by Speechiness")
top_speechiness_tracks = df_filtered.nlargest(10, 'speechiness')[['song_title', 'artist', 'speechiness']]
st.write(top_speechiness_tracks)

st.info("✨ This dashboard covers all major Spotify EDA insights. Use the sidebar to filter by artist and compare multiple feature distributions interactively!")
