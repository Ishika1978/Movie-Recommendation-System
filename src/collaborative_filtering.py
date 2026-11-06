
import pandas as pd
from sklearn.neighbors import NearestNeighbors


class CollaborativeFiltering:

    def __init__(self, n_neighbors=10):

        self.n_neighbors = n_neighbors
        self.model = None
        self.user_movie_matrix = None
        self.movies = None

    def fit(self, ratings, movies):

        """Train collaborative filtering model."""

        self.movies = movies.copy()

        # Create movie-user rating matrix
        self.user_movie_matrix = ratings.pivot_table(
            index="movieId",
            columns="userId",
            values="rating"
        ).fillna(0)

        # Train KNN using cosine distance
        self.model = NearestNeighbors(
            metric="cosine",
            algorithm="brute"
        )

        self.model.fit(self.user_movie_matrix)

        print("Collaborative Filtering model trained!")
        print("Movies:", len(self.user_movie_matrix))
        print("Users:", len(self.user_movie_matrix.columns))

        return self

    def recommend(self, movie_id, n=10):

        """Find movies with similar user-rating patterns."""

        if self.model is None:
            raise RuntimeError("Train the model using fit() first.")

        if movie_id not in self.user_movie_matrix.index:
            return []

        movie_position = self.user_movie_matrix.index.get_loc(
            movie_id
        )

        number_of_neighbors = min(
            n + 1,
            len(self.user_movie_matrix)
        )

        distances, indices = self.model.kneighbors(
            self.user_movie_matrix.iloc[
                [movie_position]
            ],
            n_neighbors=number_of_neighbors
        )

        recommendations = []

        for distance, index in zip(
            distances.flatten(),
            indices.flatten()
        ):

            recommended_id = self.user_movie_matrix.index[index]

            # Exclude selected movie
            if recommended_id == movie_id:
                continue

            movie_data = self.movies[
                self.movies["movieId"] == recommended_id
            ]

            if movie_data.empty:
                continue

            movie = movie_data.iloc[0]

            recommendations.append({

                "movieId": int(recommended_id),

                "title": movie["title"],

                "genres": movie["genres"],

                "similarity_score": round(
                    float(1 - distance), 4
                )

            })

            if len(recommendations) >= n:
                break

        return recommendations


if __name__ == "__main__":

    from src.data_preprocessing import load_and_preprocess_data

    movies, ratings = load_and_preprocess_data()

    recommender = CollaborativeFiltering()

    recommender.fit(ratings, movies)

    results = recommender.recommend(
        movie_id=1,
        n=10
    )

    print("\nTop 10 Collaborative Recommendations:\n")

    for movie in results:

        print(
            movie["title"],
            "|",
            movie["genres"],
            "| Similarity:",
            movie["similarity_score"]
        )
