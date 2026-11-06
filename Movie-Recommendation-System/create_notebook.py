
import nbformat as nbf
import os

# Create a new Jupyter Notebook
notebook = nbf.v4.new_notebook()

cells = []

def markdown(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def code(text):
    cells.append(nbf.v4.new_code_cell(text))


# ------------------------------------------------
# 1. PROJECT INTRODUCTION
# ------------------------------------------------

markdown("""
# Movie Recommendation System

### Machine Learning Internship Project

**Objective:** Build a movie recommendation system using Machine Learning.

### Algorithms Used
1. Content-Based Filtering
2. Collaborative Filtering
3. K-Nearest Neighbors (KNN)
4. Cosine Similarity

**Dataset:** MovieLens Latest Small Dataset
""")


# ------------------------------------------------
# 2. IMPORT LIBRARIES
# ------------------------------------------------

markdown("## 1. Import Required Libraries")

code("""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

import warnings
warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid")

print("All libraries imported successfully!")
""")


# ------------------------------------------------
# 3. LOAD DATASET
# ------------------------------------------------

markdown("## 2. Load MovieLens Dataset")

code("""
# Find project root whether notebook runs from root or notebooks/
current = Path.cwd()

if (current / "dataset").exists():
    PROJECT_ROOT = current
elif (current.parent / "dataset").exists():
    PROJECT_ROOT = current.parent
else:
    raise FileNotFoundError(
        "Cannot find dataset folder. Run the notebook from the project "
        "root or notebooks folder."
    )

DATASET_PATH = PROJECT_ROOT / "dataset"

movies = pd.read_csv(DATASET_PATH / "movies.csv")
ratings = pd.read_csv(DATASET_PATH / "ratings.csv")

print("Movies shape:", movies.shape)
print("Ratings shape:", ratings.shape)

display(movies.head())
display(ratings.head())
""")


# ------------------------------------------------
# 4. DATA INFORMATION
# ------------------------------------------------

markdown("## 3. Dataset Information")

code("""
print("Movies Information:")
movies.info()

print("\\nRatings Information:")
ratings.info()

print("\\nMissing Values in Movies:")
display(movies.isnull().sum())

print("\\nMissing Values in Ratings:")
display(ratings.isnull().sum())

print("\\nNumber of unique movies:", movies["movieId"].nunique())
print("Number of unique users:", ratings["userId"].nunique())
""")


# ------------------------------------------------
# 5. DATA PREPROCESSING
# ------------------------------------------------

markdown("## 4. Data Preprocessing")

code("""
# Remove duplicate records
movies = movies.drop_duplicates(subset="movieId").copy()
ratings = ratings.drop_duplicates(
    subset=["userId", "movieId"]
).copy()

# Remove missing values
movies = movies.dropna(subset=["movieId", "title", "genres"])
ratings = ratings.dropna(
    subset=["userId", "movieId", "rating"]
)

# Clean genre information
movies["genres"] = movies["genres"].replace(
    "(no genres listed)", ""
)

# Extract release year
movies["year"] = movies["title"].str.extract(
    r"\\((\\d{4})\\)"
)

# Prepare content features
movies["content"] = (
    movies["genres"].str.replace("|", " ", regex=False)
)

print("Data preprocessing completed!")
print("Clean movies:", movies.shape)
print("Clean ratings:", ratings.shape)

display(movies.head())
""")


# ------------------------------------------------
# 6. EXPLORATORY DATA ANALYSIS
# ------------------------------------------------

markdown("## 5. Exploratory Data Analysis")

code("""
# Genre distribution
genre_data = movies.assign(
    genre=movies["genres"].str.split("|")
).explode("genre")

genre_data = genre_data[
    genre_data["genre"].notna() &
    (genre_data["genre"] != "")
]

plt.figure(figsize=(12, 6))
genre_data["genre"].value_counts().head(15).plot(
    kind="bar"
)

plt.title("Top 15 Movie Genres")
plt.xlabel("Genre")
plt.ylabel("Number of Movies")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
""")

code("""
# Rating distribution
plt.figure(figsize=(8, 5))

sns.countplot(
    data=ratings,
    x="rating"
)

plt.title("Distribution of Movie Ratings")
plt.xlabel("Rating")
plt.ylabel("Number of Users")

plt.show()

print("Average movie rating:", ratings["rating"].mean())
""")


# ------------------------------------------------
# 7. POPULAR MOVIES
# ------------------------------------------------

markdown("## 6. Popular Movie Analysis")

code("""
movie_stats = ratings.groupby("movieId").agg(
    average_rating=("rating", "mean"),
    rating_count=("rating", "count")
).reset_index()

# Require at least 20 ratings for popular movie ranking
popular_movies = movie_stats[
    movie_stats["rating_count"] >= 20
].merge(
    movies[["movieId", "title", "genres"]],
    on="movieId"
)

popular_movies = popular_movies.sort_values(
    ["average_rating", "rating_count"],
    ascending=[False, False]
)

print("Top 10 Popular Movies:")
display(popular_movies.head(10))
""")


# ------------------------------------------------
# 8. CONTENT BASED FILTERING
# ------------------------------------------------

markdown("## 7. Content-Based Filtering Using TF-IDF")

code("""
tfidf = TfidfVectorizer(
    stop_words="english"
)

tfidf_matrix = tfidf.fit_transform(
    movies["content"].fillna("")
)

print("TF-IDF matrix shape:", tfidf_matrix.shape)
print("Content-Based Filtering completed!")
""")


# ------------------------------------------------
# 9. COSINE SIMILARITY
# ------------------------------------------------

markdown("## 8. Cosine Similarity")

code("""
cosine_sim = cosine_similarity(
    tfidf_matrix,
    dense_output=False
)

movie_indices = pd.Series(
    movies.index,
    index=movies["title"]
).drop_duplicates()

def recommend_movies(movie_title, n=10):

    if movie_title not in movie_indices.index:
        print("Movie not found!")
        return pd.DataFrame()

    idx = movie_indices[movie_title]

    scores = cosine_sim[idx].toarray().flatten()

    similar_indices = np.argsort(scores)[::-1]

    similar_indices = [
        i for i in similar_indices if i != idx
    ][:n]

    result = movies.iloc[similar_indices][
        ["movieId", "title", "genres"]
    ].copy()

    result["similarity_score"] = scores[similar_indices]

    return result.reset_index(drop=True)


print("Recommendations for Toy Story:")
display(recommend_movies("Toy Story (1995)", 10))
""")


# ------------------------------------------------
# 10. COLLABORATIVE FILTERING
# ------------------------------------------------

markdown("## 9. Collaborative Filtering Using KNN")

code("""
# Create user-movie rating matrix
user_movie_matrix = ratings.pivot_table(
    index="movieId",
    columns="userId",
    values="rating"
).fillna(0)

print("User-Movie Matrix:", user_movie_matrix.shape)

# Train KNN model
knn_model = NearestNeighbors(
    metric="cosine",
    algorithm="brute"
)

knn_model.fit(user_movie_matrix)

print("KNN model trained successfully!")
""")


# ------------------------------------------------
# 11. KNN RECOMMENDATIONS
# ------------------------------------------------

markdown("## 10. KNN Movie Recommendations")

code("""
def knn_recommendations(movie_id, n=10):

    if movie_id not in user_movie_matrix.index:
        print("Movie ID not found!")
        return pd.DataFrame()

    movie_position = user_movie_matrix.index.get_loc(
        movie_id
    )

    distances, indices = knn_model.kneighbors(
        user_movie_matrix.iloc[[movie_position]],
        n_neighbors=min(n + 1, len(user_movie_matrix))
    )

    recommended_ids = user_movie_matrix.index[
        indices.flatten()
    ]

    similarities = 1 - distances.flatten()

    result = pd.DataFrame({
        "movieId": recommended_ids,
        "similarity_score": similarities
    })

    result = result[
        result["movieId"] != movie_id
    ]

    result = result.merge(
        movies[["movieId", "title", "genres"]],
        on="movieId"
    )

    return result.head(n).reset_index(drop=True)


print("Movies similar to Toy Story:")
display(knn_recommendations(1, 10))
""")


# ------------------------------------------------
# 12. MODEL EVALUATION
# ------------------------------------------------

markdown("## 11. Model Evaluation")

code("""
# Split ratings into training and testing sets
train_ratings, test_ratings = train_test_split(
    ratings,
    test_size=0.2,
    random_state=42
)

# User-average baseline prediction
user_means = train_ratings.groupby(
    "userId"
)["rating"].mean()

global_mean = train_ratings["rating"].mean()

predictions = (
    test_ratings["userId"]
    .map(user_means)
    .fillna(global_mean)
)

mae = mean_absolute_error(
    test_ratings["rating"],
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        test_ratings["rating"],
        predictions
    )
)

print("User Average Baseline Evaluation")
print("MAE:", round(mae, 4))
print("RMSE:", round(rmse, 4))
""")

markdown("""
**Evaluation note:** MAE and RMSE above measure a simple user-average
rating prediction baseline. They do not directly measure the quality
of the content-based or KNN recommendation rankings.
""")


# ------------------------------------------------
# 13. SAVE MODELS
# ------------------------------------------------

markdown("## 12. Save Trained Models")

code("""
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

joblib.dump(
    movies,
    MODEL_DIR / "movie_data.pkl"
)

joblib.dump(
    tfidf,
    MODEL_DIR / "tfidf_vectorizer.pkl"
)

joblib.dump(
    cosine_sim,
    MODEL_DIR / "similarity_matrix.pkl"
)

joblib.dump(
    knn_model,
    MODEL_DIR / "knn_model.pkl"
)

joblib.dump(
    user_movie_matrix,
    MODEL_DIR / "user_movie_matrix.pkl"
)

print("All models saved successfully!")
print("Model directory:", MODEL_DIR)
""")


# ------------------------------------------------
# 14. CONCLUSION
# ------------------------------------------------

markdown("""
## 13. Project Conclusion

The Movie Recommendation System was developed using the MovieLens
dataset and Python Machine Learning libraries.

### Implemented Techniques

1. Content-Based Filtering
2. Collaborative Filtering
3. K-Nearest Neighbors
4. Cosine Similarity

### Key Features

- Movie data preprocessing
- Exploratory data analysis
- Popular movie analysis
- Movie similarity calculation
- Personalized recommendation techniques
- Model saving

### Future Enhancements

- Add movie posters.
- Build a Streamlit web application.
- Add user login and preferences.
- Implement Matrix Factorization.
- Evaluate recommendations using Precision@K and Recall@K.

**Project Status:** Initial Machine Learning notebook completed.
""")


# ------------------------------------------------
# 15. CREATE NOTEBOOK FILE
# ------------------------------------------------

notebook["cells"] = cells

notebook["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "name": "python"
    }
}

output_folder = "notebooks"
os.makedirs(output_folder, exist_ok=True)

output_file = os.path.join(
    output_folder,
    "movie_recommendation_analysis.ipynb"
)

nbf.write(notebook, output_file)

print("\\n====================================")
print("NOTEBOOK CREATED SUCCESSFULLY!")
print("File:", output_file)
print("Total cells:", len(cells))
print("====================================")
