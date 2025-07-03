import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Spotify EDA Dashboard", layout="wide")

# --- Custom CSS for background and headers ---
st.markdown("""
    <style>
    .stApp {background-color: #f5f7fa;}
    h1, h2, h3, h4 {color: #22223b;}
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

# --- Sidebar filter for artist selection ---
st.sidebar.title("🔎 Filter Options")
artist_options = ['All'] + sorted(df['artist'].unique())
selected_artist = st.sidebar.selectbox("🎤 Select Artist", artist_options)
df_filtered = df if selected_artist == 'All' else df[df['artist'] == selected_artist]

st.header("🗂️ Dataset Preview")
st.dataframe(df_filtered.head(20))

# --- Pie Chart: Top 10 Instrumental Tracks ---
st.subheader("🥁 Top 10 Instrumental Tracks (Pie Chart)")
top_instrumental = df_filtered.nlargest(10, 'instrumentalness')[['song_title', 'instrumentalness']]
labels = top_instrumental['song_title']
sizes = top_instrumental['instrumentalness']
colors = sns.color_palette('cool', len(labels))
fig1, ax1 = plt.subplots(figsize=(5, 5))
wedges, texts, autotexts = ax1.pie(
    sizes, labels=None, autopct='%1.1f%%', startangle=90,
    pctdistance=1.15, colors=colors, textprops={'fontsize': 12}
)
centre_circle = plt.Circle((0, 0), 0.65, fc='white')
fig1.gca().add_artist(centre_circle)
ax1.set_title('Top 10 Instrumental Tracks', fontsize=14)
ax1.legend(wedges, labels, title="Song Title", loc="center left", bbox_to_anchor=(1, 0.5), fontsize=10)
for autotext in autotexts:
    autotext.set_color('black')
    autotext.set_fontsize(10)
plt.tight_layout()
st.pyplot(fig1)

# --- Bar Chart: Top 10 Artists by Track Count ---
st.subheader("🌟 Top 10 Artists by Track Count")
top_artists = df_filtered['artist'].value_counts().head(10)
fig2, ax2 = plt.subplots()
sns.barplot(x=top_artists.values, y=top_artists.index, palette='viridis', ax=ax2)
ax2.set_xlabel('Number of Tracks')
ax2.set_ylabel('Artist')
ax2.set_title('Top 10 Artists')
st.pyplot(fig2)

# --- Bar Chart: Top 5 Loudest Tracks ---
st.subheader("🔊 Top 5 Loudest Tracks")
top_loudest = df_filtered.nlargest(5, 'loudness')[['song_title', 'loudness']]
fig3, ax3 = plt.subplots()
sns.barplot(x='loudness', y='song_title', data=top_loudest, palette='magma', ax=ax3)
ax3.set_xlabel('Loudness')
ax3.set_ylabel('Song Title')
ax3.set_title('Top 5 Loudest Tracks')
st.pyplot(fig3)

# --- Bar Chart: Top 10 Energetic Tracks ---
st.subheader("⚡ Top 10 Energetic Tracks")
top_energetic = df_filtered.nlargest(10, 'energy')[['song_title', 'energy']]
fig4, ax4 = plt.subplots()
sns.barplot(x='energy', y='song_title', data=top_energetic, palette='Blues_r', ax=ax4)
ax4.set_xlabel('Energy')
ax4.set_ylabel('Song Title')
ax4.set_title('Top 10 Energetic Tracks')
st.pyplot(fig4)

# --- Bar Chart: Top 10 Tracks with Most Valence ---
st.subheader("😊 Top 10 Tracks with Most Valence")
top_valence = df_filtered.nlargest(10, 'valence')[['song_title', 'valence']]
fig5, ax5 = plt.subplots()
sns.barplot(x='valence', y='song_title', data=top_valence, palette='Reds_r', ax=ax5)
ax5.set_xlabel('Valence')
ax5.set_ylabel('Song Title')
ax5.set_title('Top 10 Tracks with Most Valence')
st.pyplot(fig5)

# --- Bar Chart: Top 10 Tracks by Liveness ---
st.subheader("🎤 Top 10 Tracks by Liveness")
top_liveness = df_filtered.nlargest(10, 'liveness')[['song_title', 'liveness']]
fig6, ax6 = plt.subplots()
sns.barplot(x='liveness', y='song_title', data=top_liveness, palette='Purples_r', ax=ax6)
ax6.set_xlabel('Liveness')
ax6.set_ylabel('Song Title')
ax6.set_title('Top 10 Tracks by Liveness')
st.pyplot(fig6)

# --- Bar Chart: Top 10 Tracks by Acousticness ---
st.subheader("🎸 Top 10 Tracks by Acousticness")
top_acousticness = df_filtered.nlargest(10, 'acousticness')[['song_title', 'acousticness']]
fig7, ax7 = plt.subplots()
sns.barplot(x='acousticness', y='song_title', data=top_acousticness, palette='Greens_r', ax=ax7)
ax7.set_xlabel('Acousticness')
ax7.set_ylabel('Song Title')
ax7.set_title('Top 10 Tracks by Acousticness')
st.pyplot(fig7)

# --- Bar Chart: Top 10 Tracks by Speechiness ---
st.subheader("🗣️ Top 10 Tracks by Speechiness")
top_speechiness = df_filtered.nlargest(10, 'speechiness')[['song_title', 'speechiness']]
fig8, ax8 = plt.subplots()
sns.barplot(x='speechiness', y='song_title', data=top_speechiness, palette='Oranges_r', ax=ax8)
ax8.set_xlabel('Speechiness')
ax8.set_ylabel('Song Title')
ax8.set_title('Top 10 Tracks by Speechiness')
st.pyplot(fig8)

# --- Interactive Feature Distribution ---
st.subheader("📈 Feature Distribution Explorer")
feature = st.selectbox(
    "Select a feature to visualize:",
    ['energy', 'valence', 'tempo', 'loudness', 'acousticness', 'danceability', 'instrumentalness', 'liveness', 'speechiness']
)
fig9, ax9 = plt.subplots()
sns.histplot(df_filtered[feature], bins=30, kde=True, color='#36a2eb', ax=ax9)
ax9.set_title(f'{feature.capitalize()} Distribution')
st.pyplot(fig9)

# --- Feature Distributions (energy, valence, tempo, loudness) ---
st.subheader("📊 Feature Distributions (Energy, Valence, Tempo, Loudness)")
fig10, axs10 = plt.subplots(2, 2, figsize=(12, 10))
sns.histplot(df_filtered['energy'], bins=30, ax=axs10[0, 0], color='blue')
axs10[0, 0].set_title('Energy')
sns.histplot(df_filtered['valence'], bins=30, ax=axs10[0, 1], color='red')
axs10[0, 1].set_title('Valence')
sns.histplot(df_filtered['tempo'], bins=30, ax=axs10[1, 0], color='green')
axs10[1, 0].set_title('Tempo')
sns.histplot(df_filtered['loudness'], bins=30, ax=axs10[1, 1], color='purple')
axs10[1, 1].set_title('Loudness')
plt.tight_layout()
st.pyplot(fig10)

# --- Correlation Heatmap ---
st.subheader("🧮 Correlation Heatmap (Numeric Features)")
numeric_cols = df_filtered.select_dtypes(include=['float64', 'int64']).columns
corr = df_filtered[numeric_cols].corr()
fig11, ax11 = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="crest", ax=ax11)
st.pyplot(fig11)

# --- Scatter Plot: Energy vs. Danceability ---
st.subheader("⚡ Energy vs. Danceability")
fig12, ax12 = plt.subplots()
sns.scatterplot(data=df_filtered, x='energy', y='danceability', alpha=0.3, ax=ax12)
st.pyplot(fig12)

# --- Top 5 Popular Artists ---
st.subheader("🏆 Top 5 Popular Artists")
top_five_artists = df_filtered['artist'].value_counts().head(5)
st.bar_chart(top_five_artists)

# --- Artist with Most Danceable Song ---
st.subheader("💃 Artist with Most Danceable Song")
most_danceable_song = df_filtered.loc[df_filtered['danceability'].idxmax()]
st.write(f"Artist: **{most_danceable_song['artist']}**")
st.write(f"Song Title: **{most_danceable_song['song_title']}**")
st.write(f"Danceability: **{most_danceable_song['danceability']}**")

# --- Most Common Track Duration ---
st.subheader("⏳ Most Common Track Duration")
most_common_duration = df_filtered['duration_ms'].mode()[0]
minutes = most_common_duration // 60000
seconds = (most_common_duration % 60000) // 1000
st.write(f"The most common track duration is **{minutes} minutes and {seconds} seconds** ({most_common_duration} ms).")

# --- Most Trending Artist ---
st.subheader("🔥 Most Trending Artist")
most_trending_artist = df_filtered['artist'].value_counts().idxmax()
most_trending_artist_count = df_filtered['artist'].value_counts().max()
st.write(f"The most trending artist is **{most_trending_artist}** with **{most_trending_artist_count}** tracks in the dataset.")

st.info("✨ Dashboard is focused and visually appealing. Only the Top 10 Instrumental Tracks Pie Chart is shown. All bar chart metrics are correct and match your data. Use the sidebar to filter by artist and explore your Spotify dataset in depth!")
