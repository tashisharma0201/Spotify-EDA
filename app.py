import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Spotify EDA", layout="wide")
st.title("Spotify Exploratory Data Analysis Dashboard")

@st.cache_data
def load_data():
    df = pd.read_csv("data.csv")
    df.columns = df.columns.str.strip()
    return df

df = load_data()

st.header("Dataset Preview")
st.dataframe(df.head())

st.write("Columns in the dataset:", df.columns.tolist())

# Distribution of Danceability
st.subheader("Distribution of Danceability")
fig1, ax1 = plt.subplots()
sns.histplot(df['danceability'], bins=30, ax=ax1, color='green')
ax1.set_xlabel('Danceability')
ax1.set_ylabel('Count')
st.pyplot(fig1)

# Top 10 Artists by Track Count
st.subheader("Top 10 Artists by Track Count")
top_artists = df['artist'].value_counts().head(10)
st.bar_chart(top_artists)

# Distribution of Duration
st.subheader("Distribution of Track Duration (ms)")
fig2, ax2 = plt.subplots()
sns.histplot(df['duration_ms'], bins=30, ax=ax2, color='orange')
ax2.set_xlabel('Duration (ms)')
ax2.set_ylabel('Count')
st.pyplot(fig2)

# Correlation Heatmap
st.subheader("Correlation Heatmap (Numeric Features)")
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
corr = df[numeric_cols].corr()
fig3, ax3 = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="crest", ax=ax3)
st.pyplot(fig3)

# Energy vs. Danceability
st.subheader("Energy vs. Danceability")
fig4, ax4 = plt.subplots()
sns.scatterplot(data=df, x='energy', y='danceability', alpha=0.3, ax=ax4)
st.pyplot(fig4)

st.info("Explore more insights by extending this dashboard with additional plots and filters!")
