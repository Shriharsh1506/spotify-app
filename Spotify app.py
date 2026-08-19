#!/usr/bin/env python
# coding: utf-8

# # Phase 1: Project Introduction

# # 🎵 Spotify Music Analysis using Python
# 
# ## 📌 Project Overview
# 
# The **Spotify Music Analysis** project aims to explore and analyze a comprehensive dataset of Spotify tracks using Python and various data analysis libraries. The dataset contains detailed information about songs, artists, albums, genres, and audio characteristics, allowing us to uncover meaningful insights into music trends and listener preferences.
# 
# This project follows the complete **Data Analysis workflow**, including data cleaning, preprocessing, exploratory data analysis (EDA), visualization, and insight generation. By examining various musical attributes such as **popularity, danceability, energy, loudness, tempo, valence, acousticness, and genre**, we can identify relationships between song characteristics and their popularity.
# 
# The project also demonstrates essential data science skills such as handling missing values, removing duplicate records, feature engineering, statistical analysis, correlation analysis, and creating informative visualizations using Python libraries.
# 
# ---
# 
# # 🎯 Project Objectives
# 
# - Understand the structure and quality of the Spotify dataset.
# - Clean the dataset by handling missing values and duplicate records.
# - Perform Exploratory Data Analysis (EDA) to discover hidden patterns.
# - Analyze song popularity across different genres and artists.
# - Study relationships between audio features and popularity.
# - Identify the most popular artists, albums, and genres.
# - Explore distributions of musical features such as danceability, energy, tempo, loudness, and valence.
# - Generate meaningful business insights through visualizations.
# - Develop a well-documented data analysis project suitable for a portfolio.
# 
# ---
# 
# # 📂 Dataset Description
# In[3]:
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 

# Prevent running notebook exploratory cells during Streamlit import/run
RUN_NOTEBOOK = False

# Ensure `df` exists to avoid NameError when notebook cells reference it.
try:
    df = pd.read_csv("spotify.csv")
except Exception:
    df = pd.DataFrame()

if RUN_NOTEBOOK:
    # load the raw CSV for notebook-style exploration (already attempted above)
    df.head()


# In[4]:
    df.tail()


# In[5]:
    df.info()


# In[6]:
    df.describe()


# In[7]:
    df.columns


# In[8]:
    df.shape


# In[9]:
    df.shape[0] # rows


# In[10]:
    df.shape[1] # columns


    # # Phase 5: Data Cleaning

    # In[11]:

    df.isnull().sum()


    # In[12]:

    df.dropna(inplace=True)


    # In[13]:

    # checking whether the null vaies still exists or not 

    df.isnull().sum()


    # In[14]:

    df.duplicated().sum()


    # In[15]:

    df.drop_duplicates(inplace=True)


    # In[16]:

    df.shape[0]


    # In[17]:

    df.dtypes # checking the datatypes 


    # In[18]:

    # Check object (text) columns

    df.select_dtypes(include="object").columns


    # In[19]:

    # Remove leading and trailing spaces

    text_cols = df.select_dtypes(include="object").columns

    for col in text_cols:
        df[col] = df[col].str.strip()


    # In[20]:

    # Convert multiple spaces into a single space

    for col in text_cols:
        df[col] = df[col].str.replace(r"\s+", " ", regex=True)


    # In[21]:

    # Standardize text case
    df["track_name"] = df["track_name"].str.title()
    df["album_name"] = df["album_name"].str.title()
    df["artists"] = df["artists"].str.title()
    df["track_genre"] = df["track_genre"].str.lower()


    # In[22]:

    # Checking for unwanted spaces again

    df[text_cols].head()


    # In[23]:

    # Check for duplicate artist names caused by spacing

    df["artists"].value_counts().head(10)


    # In[24]:

    # Finding rows with empty strings

    for col in text_cols:
        print(col, (df[col] == "").sum())


    # In[25]:

    # Removing only unwanted leading/trailing symbols
    df["artists"].sample(20)


    # In[26]:

    text_cols = df.select_dtypes(include="object").columns

    for col in text_cols:
        df[col] = (
            df[col]
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
            .str.replace(";", ", ", regex=False)
            .str.replace(r"^[^\w]+|[^\w]+$", "", regex=True)
        )


    # # Phase 6: Feature Engineering

    # In[27]:

    df['duration_min']=df['duration_ms']/60000 



    # Song duration was converted from milliseconds to minutes to make the analysis easier to understand.

    # # Phase 7: Univariate Analysis

    # ## Popularity Distribution

    # In[28]:

    sns.histplot(df['popularity'],kde=True,bins=30)
    plt.title("Distribution of Song Popularity",size=15)
    plt.xlabel("Popularity")
    plt.ylabel("Number of Songs")
    plt.show()


    # **Observation:** Most songs have low to medium popularity scores. The average popularity is around 33, and the median popularity is 35. Very few songs reach very high popularity scores close to 100.

    # ## Duration Distribution 

    # In[29]:

    sns.histplot(df['duration_min'])
    plt.xlim(0,10)
    plt.title("Duration Distribution of Songs",size=15)
    plt.xlabel("Duration (Minutes)")
    plt.ylabel("Number of Songs")
    plt.show()


    # **Observation:** Most songs are between 2 and 5 minutes long, with the median duration around 3.5 minutes. Very long songs are rare and appear as outliers.

    # ## Explicit vs Non-Explicit Songs

    # In[30]:

    plt.figure(figsize=(7,5))

    ax = sns.countplot(
        data=df,
        x="explicit",
        hue="explicit",
        palette="Set2",
        legend=False
    )

    plt.title("Explicit vs Non-Explicit Songs")
    plt.xlabel("Song Type")
    plt.ylabel("Count")

    plt.xticks(
        [0,1],
        ["Non-Explicit","Explicit"]
    )

    plt.show()


    # **Observation:** Non-explicit songs are much more common than explicit songs. The dataset contains about 103,831 non-explicit songs and 9,718 explicit songs.

    # ## Genre Distribution (Top 10)

    # In[31]:

    plt.figure(figsize=(12,6))
    sns.countplot(
        y="track_genre",
        data=df,
        order=df["track_genre"].value_counts().head(10).index
    )
    plt.title("Genre Distribution")
    plt.xlabel("Count")
    plt.ylabel("Genre")
    plt.show()


    # **Observation:** The dataset is fairly balanced across many genres, with several genres having around 1,000 songs. The graph shows the genres with the highest number of tracks.

    # # Phase 8: Artist Analysis

    # ## Top 10 Artists by Number of Songs

    # In[32]:

    top_artists=(df['artists'].value_counts().head(10))
    plt.figure(figsize=(12,6))

    sns.barplot(
        x=top_artists.values,
        y=top_artists.index
    )

    plt.title("Top 10 Artists by Number of Songs")
    plt.xlabel("Number of Songs")
    plt.ylabel("Artist")
    plt.show()


    # **Observation:** The Beatles, George Jones, Stevie Wonder, Linkin Park, and Ella Fitzgerald are among the artists with the highest number of songs in the dataset

    # ## Top 10 Artist by Average Popularity

    # In[33]:

    artist_popularity = (
        df.groupby("artists")["popularity"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )
    plt.figure(figsize=(10,6))
    sns.barplot(x=artist_popularity.values,y=artist_popularity.index)
    plt.title("Top 10 Artists by Average Popularity")
    plt.show()


    # **Observation:** Some artists have very high average popularity, including Sam Smith and Kim Petras, Bizarrap and Quevedo, Manuel Turizo, and Bad Bunny. However, average popularity can be affected when an artist has only a small number of songs in the dataset.

    # # Phase 9: Genre Analysis

    # ## Top 10 Genres by Average Popularity

    # In[34]:

    genre_popularity = (
        df.groupby("track_genre")["popularity"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )
    plt.figure(figsize=(10,6))

    sns.barplot(
        x=genre_popularity.values,
        y=genre_popularity.index
    )
    plt.title("Top 10 Genres by Average Popularity")
    plt.xlabel("Average Popularity")
    plt.ylabel("Genre")
    plt.show()


    # **Observation:** Pop-film, k-pop, chill, sad, and grunge are among the genres with the highest average popularity. This shows that popularity differs noticeably across genres.

    # ## Top 10 Genres by Average Danceability

    # In[35]:

    genre_danceability=(
        df.groupby("track_genre")["danceability"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )

    plt.figure(figsize=(10,6))
    sns.barplot(x=genre_danceability.values,y=genre_danceability.index)
    plt.title("Top 10 Genres by Average Danceability")
    plt.xlabel("Average Danceability")
    plt.ylabel("Genre")
    plt.show()


    # **Observation:** Kids, chicago-house, reggaeton, latino, and reggae have the highest average danceability. These genres are generally more rhythm-focused and dance-friendly.

    # ## Top 10 Genres by Average Energy

    # In[36]:

    genre_energy=(
        df.groupby("track_genre")["energy"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )

    plt.figure(figsize=(10,6))
    sns.barplot(x=genre_energy.values,y=genre_energy.index)
    plt.title("Top 10 Genres by Average Energy")
    plt.xlabel("Average Energy")
    plt.ylabel("Genre")
    plt.show()


    # **Observation:** Death-metal, grindcore, metalcore, happy, and hardstyle have the highest average energy. This matches the intense and fast nature of these genres.

    # # Phase 10: Bivariate Analysis

    # ## Popularity vs Danceability

    # In[37]:

    plt.figure(figsize=(8,6))
    sns.scatterplot(
        x="danceability",
        y="popularity",
        data=df,
        alpha=0.5
    )
    plt.title("Popularity vs Danceability")
    plt.xlabel("Danceability")
    plt.ylabel("Popularity")
    plt.show()


    # **Observation:** Danceability has only a weak relationship with popularity. Highly danceable songs are not always highly popular.

    # ## Popularity vs Energy

    # In[38]:

    plt.figure(figsize=(8,6))
    sns.scatterplot(
        x='energy',
        y='popularity',
        data=df,
        alpha=0.5
    )
    plt.title("Popularity vs Energy")
    plt.xlabel("Energy")
    plt.ylabel("Popularity")
    plt.show()


    # **Observation:** Energy has almost no direct relationship with popularity. High-energy songs do not necessarily receive higher popularity scores.

    # ## Popularity vs Loudness

    # In[39]:

    plt.figure(figsize=(8,6))
    sns.scatterplot(
        x='loudness',
        y='popularity',
        data=df,
        alpha=0.5
    )

    plt.title("Popularity vs Loudness")
    plt.xlabel("Loudness")
    plt.ylabel("Popularity")
    plt.show()


    # **Observation:** Loudness has a very weak positive relationship with popularity. Louder songs are not strongly more popular.

    # ## Popularity vs Acousticness

    # In[40]:

    plt.figure(figsize=(8,6))
    sns.scatterplot(
        x='acousticness',
        y='popularity',
        data=df,
        alpha=0.5
    )
    plt.title("Popularity vs Acousticness")
    plt.xlabel("Acousticness")
    plt.ylabel("Popularity")
    plt.show()


    # **Observation:** Acousticness has a weak negative relationship with popularity. More acoustic songs are not necessarily more popular.

    # # Phase 11: Boxplot Analysis

    # ## Popularity Distribution by Explicit Status

    # In[41]:

    plt.figure(figsize=(8,5))
    sns.boxplot(x = 'explicit', y = 'popularity', data = df)
    plt.title("Popularity Distribution by Explicit Status")
    plt.xlabel("Explicit")
    plt.ylabel("Popularity")
    plt.show()


    # **Observation:** Explicit songs have a slightly higher median popularity than non-explicit songs. However, non-explicit songs are much more common in the dataset.

    # ## Popularity Distribution by Top 10 Genres

    # In[42]:

    top_genres = (
        df.groupby("track_genre")["popularity"]
          .mean()
          .sort_values(ascending=False)
          .head(10)
          .index
    )
    plt.figure(figsize=(8,6))
    sns.boxplot(x='track_genre',y='popularity',data=df[df['track_genre'].isin(top_genres)])
    plt.title("Popularity Distribution by Top 10 Genres")
    plt.xlabel("Genre")
    plt.ylabel("Popularity")
    plt.xticks(rotation=90)
    plt.show()


    # **Observation:** Popularity varies across genres. Some genres have higher median popularity and wider popularity ranges, showing that genre can influence popularity patterns.

    # # Phase 12: Correlation Analysis

    # In[43]:

    numeric_df = df.select_dtypes(include="number")

    plt.figure(figsize=(12,10))

    sns.heatmap(
        numeric_df.corr(),
        annot=True,
        cmap="coolwarm"
    )

    plt.title("Correlation Heatmap")
    plt.show()


    # **Observation:** The heatmap shows that no single audio feature has a strong correlation with popularity. Instrumentalness has the strongest negative relationship with popularity, but it is still weak. This suggests that popularity depends on multiple factors beyond audio features alone.

    # # Phase 13: Feature Relationships

    # In[44]:

    sample_df = df[
        ["popularity", "danceability", "energy", "tempo", "valence"]
    ].sample(3000, random_state=42)

    sns.pairplot(sample_df)

    plt.show()


    # **Observation:** The pairplot confirms that popularity does not have a strong linear relationship with danceability, energy, tempo, or valence. The points are widely spread, which supports the correlation result.

    # # Phase 14: Outlier Detection

    # In[45]:

    numeric_df = df.select_dtypes(include="number")
    plt.figure(figsize=(12,6))
    sns.boxplot(data=numeric_df)
    plt.xticks(rotation=90)
    plt.title("Outlier Detection")
    plt.show()


    # **Observation:** Several numerical features contain outliers, especially duration, tempo, loudness, speechiness, liveness, and instrumentalness. These outliers represent songs with unusual audio characteristics.

    # # Phase 15: Advanced Analysis

    # ## Top 10 Longest Songs

    # In[46]:

    df.nlargest(10, "duration_min")


    # **Observation:** The longest songs are mostly DJ mixes, extended mixes, ambient tracks, or instrumental tracks. These songs are much longer than the normal song duration range.

    # ## Top 10 Most Popular Songs

    # In[47]:

    df.nlargest(10,"popularity")


    # **Observation:** The most popular songs include globally popular tracks such as Unholy, Quevedo: Bzrp Music Sessions, I'm Good (Blue), and La Bachata. These tracks have popularity scores close to 100.

    # ## Least Popular Songs

    # In[48]:

    df.nsmallest(10,"popularity")


    # **Observation:** The least popular songs have a popularity score of 0. This may indicate very low listener engagement, limited visibility, or older/less promoted tracks.



# ## Top Genres by Average Energy

# In[49]:


genre_energy=(df.groupby("track_genre")["energy"].mean().sort_values(ascending=False).head(10))
plt.figure(figsize=(10,6))
sns.barplot(x=genre_energy.values,
            y=genre_energy.index
           )
plt.title("Top Genres by Average Energy")
plt.xlabel("Average Energy")
plt.ylabel("Genre")
plt.show()


# **Observation:** The highest-energy genres are mostly metal, hardstyle, drum-and-bass, and other intense genres. These genres usually contain loud, fast, and powerful tracks.

# ## Top 10 Genres by Average Danceability

# In[50]:


genre_danceability=(df.groupby("track_genre")["danceability"].mean().sort_values(ascending=False).head(10))
plt.figure(figsize=(10,6))
sns.barplot(x=genre_danceability.values,y=genre_danceability.index)
plt.title("Top 10 Genres by Average Danceability")
plt.xlabel("Average Danceability")
plt.ylabel("Genre")
plt.show()


# **Observation:** The most danceable genres are mostly rhythm-based genres such as reggaeton, latino, reggae, hip-hop, dancehall, and house-related genres.

# ## Top 10 Albums by Average Popularity

# In[51]:


album_popularity = (
    df.groupby("album_name")["popularity"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(10,6))
sns.barplot(x=album_popularity.values, y=album_popularity.index)
plt.title("Top 10 Albums by Average Popularity")
plt.xlabel("Average Popularity")
plt.ylabel("Album")
plt.show()


# **Observation:** Albums such as Unholy, Quevedo: Bzrp Music Sessions, Vol. 52, La Bachata, Indigo, and I Ain't Worried have very high average popularity. However, albums with fewer tracks may get inflated average popularity scores.

# ## Songs with Highest Tempo

# In[52]:


df.nlargest(10,"tempo")


# **Observation:** The highest-tempo songs have unusually fast BPM values and appear as tempo outliers in the dataset.

# # Phase 16: Key Insights
# 
# • The dataset was successfully cleaned by removing duplicates, handling missing values, and standardizing text columns.
# 
# • Most songs have medium popularity, while only a small number achieve very high popularity.
# 
# • The majority of songs have a duration between 3 and 4 minutes.
# 
# • Non-explicit songs are more common than explicit songs in the dataset.
# 
# • Danceability, energy, tempo, and acousticness show only weak relationships with popularity.
# 
# • Explicit songs appear to have a slightly higher median popularity than non-explicit songs.
# 
# • The correlation heatmap indicates that most audio features have weak correlations with popularity, while some audio features show moderate relationships with each other.
# 
# • Several numerical features contain outliers, representing songs with unusually high or low values.
# 
# • Genre-based analysis shows that different genres have different average popularity, energy, and danceability.
# 
# • Advanced analysis identified the longest songs, most popular songs, highest tempo songs, and genres with the highest average energy and danceability.

# # Phase 17: Conclusion
# 
# This project performed Exploratory Data Analysis (EDA) on the Spotify Songs dataset using Python libraries such as Pandas, NumPy, Matplotlib, and Seaborn. The analysis included data cleaning, feature engineering, univariate analysis, bivariate analysis, correlation analysis, outlier detection, and advanced analysis.
# 
# The results show that no single audio feature strongly determines a song's popularity. Features such as danceability, energy, tempo, acousticness, and loudness show only weak relationships with popularity. This suggests that popularity is influenced by multiple factors beyond audio characteristics alone.
# 
# The project also highlighted clear differences among genres in terms of average popularity, energy, and danceability. In addition, the analysis identified interesting patterns such as the most popular songs, longest songs, highest-tempo songs, and outliers in numerical features.
# 
# Overall, this analysis provides useful insights into Spotify music characteristics and demonstrates the importance of Exploratory Data Analysis before applying advanced analytics or machine learning models.

# In[53]:


import streamlit as st


# In[55]:


# Set page configuration
st.set_page_config(
    page_title="🎵 Spotify Music Analysis", layout="wide"
)

st.title("🎵 Spotify Music Analysis Dashboard")
st.markdown("Explore audio features, genre trends, and track popularity.")


# Cache data to prevent reloading on every interaction
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("spotify_cleaned.csv")
    except FileNotFoundError:
        # fall back to original csv if cleaned file not present
        df = pd.read_csv("spotify.csv")

    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    # Clean text columns robustly
    text_cols = df.select_dtypes(include="object").columns
    for col in text_cols:
        # ensure values are strings before string operations
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].str.replace(r"\s+", " ", regex=True)
        df[col] = df[col].str.replace(";", ", ", regex=False)
        df[col] = df[col].str.replace(r"^[^\w]+|[^\w]+$", "", regex=True)

    # Feature Engineering
    df["duration_min"] = df["duration_ms"] / 60000
    return df


df = load_data()

# Optional: genre prediction utilities
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
import numpy as np

# ----------------- SIDEBAR FILTERS -----------------
st.sidebar.header("Filter Options")

# Genre Filter
genres = sorted(df["track_genre"].unique())
selected_genre = st.sidebar.multiselect(
    "Select Genre(s):", genres, default=genres[:3]
)

# Popularity Filter
min_pop, max_pop = int(df["popularity"].min()), int(df["popularity"].max())
popularity_range = st.sidebar.slider(
    "Popularity Range:", min_pop, max_pop, (min_pop, max_pop)
)

# Search box (tracks, artists, albums)
search_query = st.sidebar.text_input(
    "Search (track / artist / album):", value=""
)

# Prediction input: song name for lookup/prediction
predict_name = st.sidebar.text_input("Predict genre for song name:", value="")
predict_button = st.sidebar.button("Predict Genre")

# Apply Filters (handle empty selection)
if not selected_genre:
    filtered_df = df.copy()
else:
    filtered_df = df[
        (df["track_genre"].isin(selected_genre))
        & (df["popularity"].between(popularity_range[0], popularity_range[1]))
    ]

# Apply search filter if provided
if search_query and not filtered_df.empty:
    q = search_query.strip()
    mask = (
        filtered_df["track_name"].str.contains(q, case=False, na=False)
        | filtered_df["artists"].str.contains(q, case=False, na=False)
        | filtered_df["album_name"].str.contains(q, case=False, na=False)
    )
    filtered_df = filtered_df[mask]

# If user asks to predict genre for a song name
if predict_button and predict_name:
    q = predict_name.strip()
    # exact lookup
    exact = df[df["track_name"].str.lower() == q.lower()]
    if not exact.empty:
        st.sidebar.success(f"Exact match found — Genre: {exact.iloc[0]['track_genre']}")
    else:
        # fuzzy lookup using isin contains
        fuzzy = df[df["track_name"].str.contains(q, case=False, na=False)]
        if not fuzzy.empty:
            top = fuzzy.iloc[0]
            st.sidebar.info(f"Fuzzy match — Closest: {top['track_name']} (Genre: {top['track_genre']})")
        else:
            # Train a simple text-based classifier on track_name -> track_genre
            st.sidebar.info("No match found. Training simple name->genre classifier...")
            X = df["track_name"].astype(str)
            y = df["track_genre"].astype(str)
            # encode labels
            le = LabelEncoder()
            y_enc = le.fit_transform(y)
            X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42)
            vec = TfidfVectorizer(ngram_range=(1,2), max_features=5000)
            clf = make_pipeline(vec, RandomForestClassifier(n_estimators=100, random_state=42))
            clf.fit(X_train, y_train)
            pred = clf.predict([q])[0]
            pred_label = le.inverse_transform([pred])[0]
            st.sidebar.success(f"Predicted genre: {pred_label}")

# ----------------- MAIN DASHBOARD -----------------

# Key Metrics
if filtered_df.empty:
    st.warning("No data available for the selected filters.")
else:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Tracks", len(filtered_df))
    col2.metric("Avg Popularity", round(filtered_df["popularity"].mean(), 1))
    col3.metric("Avg Duration (min)", round(filtered_df["duration_min"].mean(), 2))
    col4.metric("Avg Tempo (BPM)", round(filtered_df["tempo"].mean(), 1))

st.markdown("---")

# Visualizations Row 1
c1, c2 = st.columns(2)

with c1:
    st.subheader("Popularity Distribution")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(filtered_df["popularity"], kde=True, bins=20, ax=ax)
    ax.set_xlabel("Popularity")
    ax.set_ylabel("Count")
    st.pyplot(fig)

with c2:
    st.subheader("Top 10 Artists in Selection")
    fig, ax = plt.subplots(figsize=(6, 4))
    top_artists = filtered_df["artists"].value_counts().head(10)
    sns.barplot(x=top_artists.values, y=top_artists.index, ax=ax)
    ax.set_xlabel("Song Count")
    st.pyplot(fig)

# Audio Features Scatter Plot
st.subheader("Danceability vs. Energy")
fig, ax = plt.subplots(figsize=(8, 4))
sns.scatterplot(
    data=filtered_df,
    x="danceability",
    y="energy",
    hue="track_genre",
    alpha=0.7,
    ax=ax,
)
st.pyplot(fig)

# Raw Data Display
if st.checkbox("Show Raw Data"):
    st.dataframe(filtered_df.head(100))


# In[57]:





# In[ ]:




