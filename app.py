import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Streamlit Page Config ---
st.set_page_config(page_title="Spotify EDA Dashboard", layout="wide")

# --- Custom CSS for User-Friendly UI ---
st.markdown(
    """
    <style>
    .stApp {background-color: #f5f7fa;}
    .main {color: #22223b;}
    h1, h2, h3, h4 {color: #22223b;}
    .css-1d391kg {color: #22223b;}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🎵 Spotify Exploratory Data Analysis Dashboard")

@st.cache_data
def load_data():
    df = pd.read_csv("data.csv")
    df.columns = df.columns.str.strip()
    return df

df = load_data()

# --- Dataset Preview ---
st.header("Dataset Preview")
st.dataframe(df.head(20))

# --- Pie Chart: Mode Distribution (Major/Minor) ---
st.subheader("Mode Distribution (Major/Minor)")
mode_counts = df['mode'].value_counts()
labels = ['Major' if x == 1 else 'Minor' for x in mode_counts.index]
fig1, ax1 = plt.subplots()
ax1.pie(mode_counts, labels=labels, autopct='%1.1f%%', colors=['#36a2eb', '#ff6384'], startangle=90, textprops={'fontsize': 12})
ax1.set_title('Mode Distribution')
st.pyplot(fig1)

# --- Pie Chart: Time Signature Distribution ---
st.subheader("Time Signature Distribution")
ts_counts = df['time_signature'].value_counts()
fig2, ax2 = plt.subplots()
ax2.pie(ts_counts, labels=ts_counts.index, autopct='%1.1f%%', colors=sns.color_palette('pastel'), startangle=90, textprops={'fontsize': 12})
ax2.set_title('Time Signature Distribution')
st.pyplot(fig2)

# --- Pie Chart: Key Distribution ---
st.subheader("Key Distribution")
key_counts = df['key'].value_counts()
fig3, ax3 = plt.subplots()
ax3.pie(key_counts, labels=key_counts.index, autopct='%1.1f%%', colors=sns.color_palette('Set3'), startangle=90, textprops={'fontsize': 12})
ax3.set_title('Key Distribution')
st.pyplot(fig3)

# --- Bar Chart: Top 10 Artists ---
st.subheader("Top 10 Artists by Track Count")
top_artists = df['artist'].value_counts().head(10)
fig4, ax4 = plt.subplots()
sns.barplot(x=top_artists.values, y=top_artists.index, palette='viridis', ax=ax4)
ax4.set_xlabel('Number of Tracks')
ax4.set_ylabel('Artist')
st.pyplot(fig4)

# --- Bar Chart: Top 5 Loudest Tracks ---
st.subheader("Top 5 Loudest Tracks")
top_loudest = df.nlargest(5, 'loudness')[['song_title', 'loudness']]
fig5, ax5 = plt.subplots()
sns.barplot(x='loudness', y='song_title', data=top_loudest, palette='magma', ax=ax5)
ax5.set_xlabel('Loudness')
ax5.set_ylabel('Song Title')
st.pyplot(fig5)

# --- Bar Chart: Top 10 Instrumental Tracks ---
st.subheader("Top 10 Instrumental Tracks")
top_instrumental = df.nlargest(10, 'instrumentalness')[['song_title', 'instrumentalness']]
fig6, ax6 = plt.subplots()
sns.barplot(x='instrumentalness', y='song_title', data=top_instrumental, palette='cool', ax=ax6)
ax6.set_xlabel('Instrumentalness')
ax6.set_ylabel('Song Title')
st.pyplot(fig6)

# --- Interactive Feature Distribution ---
st.subheader("Feature Distribution Explorer")
feature = st.selectbox(
    "Select a feature to visualize:",
    ['energy', 'valence', 'tempo', 'loudness', 'acousticness', 'danceability', 'instrumentalness', 'liveness', 'speechiness']
)
fig7, ax7 = plt.subplots()
sns.histplot(df[feature], bins=30, kde=True, color='#36a2eb', ax=ax7)
ax7.set_title(f'{feature.capitalize()} Distribution')
st.pyplot(fig7)

# --- Correlation Heatmap ---
st.subheader("Correlation Heatmap (Numeric Features)")
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
corr = df[numeric_cols].corr()
fig8, ax8 = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="crest", ax=ax8)
st.pyplot(fig8)

# --- Scatter Plot: Energy vs. Danceability ---
st.subheader("Energy vs. Danceability")
fig9, ax9 = plt.subplots()
sns.scatterplot(data=df, x='energy', y='danceability', alpha=0.3, ax=ax9)
st.pyplot(fig9)

# --- Top 5 Popular Artists ---
st.subheader("Top 5 Popular Artists")
top_five_artists = df['artist'].value_counts().head(5)
st.bar_chart(top_five_artists)

# --- Artist with Most Danceable Song ---
st.subheader("Artist with Most Danceable Song")
most_danceable_song = df.loc[df['danceability'].idxmax()]
st.write(f"Artist: **{most_danceable_song['artist']}**")
st.write(f"Song Title: **{most_danceable_song['song_title']}**")
st.write(f"Danceability: **{most_danceable_song['danceability']}**")

# --- Most Common Track Duration ---
st.subheader("Most Common Track Duration")
most_common_duration = df['duration_ms'].mode()[0]
minutes = most_common_duration // 60000
seconds = (most_common_duration % 60000) // 1000
st.write(f"The most common track duration is **{minutes} minutes and {seconds} seconds** ({most_common_duration} ms).")

# --- Most Trending Artist ---
st.subheader("Most Trending Artist")
most_trending_artist = df['artist'].value_counts().idxmax()
most_trending_artist_count = df['artist'].value_counts().max()
st.write(f"The most trending artist is **{most_trending_artist}** with **{most_trending_artist_count}** tracks in the dataset.")

# --- Top 10 Energetic Tracks ---
st.subheader("Top 10 Energetic Tracks")
top_energetic_tracks = df.nlargest(10, 'energy')[['song_title', 'artist', 'energy']]
st.write(top_energetic_tracks)

# --- Top 10 Tracks with Most Valence ---
st.subheader("Top 10 Tracks with Most Valence")
top_valence_tracks = df.nlargest(10, 'valence')[['song_title', 'artist', 'valence']]
st.write(top_valence_tracks)

# --- Top 10 Tracks by Liveness ---
st.subheader("Top 10 Tracks by Liveness")
top_liveness_tracks = df.nlargest(10, 'liveness')[['song_title', 'artist', 'liveness']]
st.write(top_liveness_tracks)

# --- Top 10 Tracks by Acousticness ---
st.subheader("Top 10 Tracks by Acousticness")
top_acousticness_tracks = df.nlargest(10, 'acousticness')[['song_title', 'artist', 'acousticness']]
st.write(top_acousticness_tracks)

# --- Top 10 Tracks by Speechiness ---
st.subheader("Top 10 Tracks by Speechiness")
top_speechiness_tracks = df.nlargest(10, 'speechiness')[['song_title', 'artist', 'speechiness']]
st.write(top_speechiness_tracks)

st.info("This dashboard covers all major Spotify EDA insights with interactive and visually engaging charts. Explore different features using the dropdown above!")

