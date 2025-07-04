# Spotify-EDA
This project performs Exploratory Data Analysis (EDA) on Spotify dataset(s) to gain insights into music trends, genres, and other characteristics. The analysis includes various visualizations and statistical summaries to understand the distribution and relationships between different musical attributes.
The dashboard is built with Streamlit and is fully interactive and user-friendly.

# How to Run This Project
1.Clone the repository:
bash
git clone https://github.com/tashisharma10/Spotify-EDA.git
cd your-repo-name

2.Install dependencies:
bash
pip install -r requirements.txt

Place your data file:
Ensure data.csv is present in the project root.

3.Run the dashboard:
bash
streamlit run app.py
The app will open in your browser at http://localhost:8501.

An interactive dashboard for Spotify dataset exploratory data analysis, built with Streamlit.
# Visit:https://spotify-eda-fyyx6kilgsbprukuhileym.streamlit.app/

# Dataset
Source: Kaggle
The dashboard uses a CSV file (data.csv) with columns like:
acousticness, danceability, duration_ms, energy, instrumentalness, key, liveness, loudness, mode, speechiness, tempo, time_signature, valence, target, song_title, artist

# Description
The dataset used in this project is a collection of Spotify track attributes that provides detailed information about various musical tracks. The dataset includes a variety of features that describe different aspects of the tracks, such as their acoustic properties, mood, and tempo.

# Features
Sidebar filter to select and explore by artist.

Visualize top artists, loudest tracks, most danceable songs, and more.

Overlay and compare feature distributions (energy, danceability, valence, etc.).

Correlation heatmap and scatter plots for feature relationships.

Summary tables for top tracks by various metrics.

Modern, emoji-enhanced, and responsive UI.

# Example Visualizations
Top 5 Popular Artists (bar chart)

Top 5 Loudest Tracks (table)

Most Danceable Song (summary)

Top 10 Instrumental Tracks (table)

Overlayed feature distributions (histograms)

Correlation heatmap

Energy vs. Danceability (scatter plot)

Top tracks by liveness, acousticness, speechiness, etc.

# Requirements
Python 3.8+
streamlit, pandas, matplotlib, seaborn (install via requirements.txt)

# File Structure
text
.
├── app.py
├── data.csv
├── requirements.txt
└── README.md
