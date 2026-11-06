
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedRecommender:

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            stop_words="english"
        )
        self.tfidf_matrix = None
        self.similarity_matrix = None
        self.movies = None
        self.movie_indices = None

    def fit(self, movies):
        """Train the content-based recommendation model."""

        self.movies = movies.copy().reset_index(drop=True)

        self.tfidf_matrix = self.vectorizer.fit_transform(
            self.movies["content"].fillna("")
        )

        self.similarity_matrix = cosine_similarity(
            self.tfidf_matrix,
            dense_output=False
        )

        self.movie_indices = {
            title: index
            for index, title in enumerate(self.movies["title"])
        }

        print("Content-Based Model trained successfully!")
        print("Movies processed:", len(self.movies))

        return self

    def recommend(self, movie_title, n=10):
        """Recommend movies similar to the selected movie."""

        if self.movies is None:
            raise RuntimeError("Train the model using fit() first.")

        if movie_title not in self.movie_indices:
            return []

        index = self.movie_indices[movie_title]

        scores = self.similarity_matrix[
            index
        ].toarray().flatten()

        similar_indices = np.argsort(scores)[::-1]

        recommendations = []

        for i in similar_indices:

            if i == index:
                continue

            movie = self.movies.iloc[i]

            recommendations.append({
                "movieId": int(movie["movieId"]),
                "title": movie["title"],
                "genres": movie["genres"],
                "similarity_score": round(
                    float(scores[i]), 4
                )
            })

            if len(recommendations) == n:
                break

        return recommendations


if __name__ == "__main__":

    from src.data_preprocessing import load_and_preprocess_data

    movies, ratings = load_and_preprocess_data()

    model = ContentBasedRecommender()
    model.fit(movies)

    results = model.recommend(
        "Toy Story (1995)",
        n=10
    )

    print("\nTop 10 Similar Movies:")

    for movie in results:
        print(
            movie["title"],
            "|",
            movie["genres"],
            "| Similarity:",
            movie["similarity_score"]
        )
