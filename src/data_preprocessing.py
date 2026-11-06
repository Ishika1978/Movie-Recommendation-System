
import pandas as pd
from pathlib import Path


# Find the main project directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = PROJECT_ROOT / "dataset"


def load_movies():
    """Load movie information from CSV."""

    file_path = DATASET_PATH / "movies.csv"

    if not file_path.exists():
        raise FileNotFoundError(
            f"Movie dataset not found: {file_path}"
        )

    movies = pd.read_csv(file_path)

    return movies


def load_ratings():
    """Load user ratings from CSV."""

    file_path = DATASET_PATH / "ratings.csv"

    if not file_path.exists():
        raise FileNotFoundError(
            f"Ratings dataset not found: {file_path}"
        )

    ratings = pd.read_csv(file_path)

    return ratings


def preprocess_movies(movies):
    """Clean and prepare movie data."""

    movies = movies.copy()

    movies.drop_duplicates(
        subset="movieId",
        inplace=True
    )

    movies.dropna(
        subset=["movieId", "title", "genres"],
        inplace=True
    )

    movies["genres"] = movies["genres"].replace(
        "(no genres listed)", ""
    )

    movies["year"] = movies["title"].str.extract(
        r"\((\d{4})\)"
    )

    movies["content"] = (
        movies["genres"]
        .str.replace("|", " ", regex=False)
    )

    return movies.reset_index(drop=True)


def preprocess_ratings(ratings):
    """Clean and prepare rating data."""

    ratings = ratings.copy()

    ratings.drop_duplicates(
        subset=["userId", "movieId"],
        inplace=True
    )

    ratings.dropna(
        subset=["userId", "movieId", "rating"],
        inplace=True
    )

    return ratings.reset_index(drop=True)


def load_and_preprocess_data():

    movies = load_movies()
    ratings = load_ratings()

    movies = preprocess_movies(movies)
    ratings = preprocess_ratings(ratings)

    print("Data loaded and preprocessed successfully!")
    print("Total movies:", len(movies))
    print("Total ratings:", len(ratings))

    return movies, ratings


if __name__ == "__main__":

    movies, ratings = load_and_preprocess_data()

    print("\nSample Movies:")
    print(movies.head())

    print("\nSample Ratings:")
    print(ratings.head())

