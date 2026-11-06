
from src.data_preprocessing import load_and_preprocess_data
from src.content_based import ContentBasedRecommender
from src.collaborative_filtering import CollaborativeFiltering


class MovieRecommendationEngine:

    def __init__(self):

        self.movies = None
        self.ratings = None

        self.content_model = ContentBasedRecommender()
        self.collaborative_model = CollaborativeFiltering()

    def train(self):

        """Load data and train both recommendation models."""

        print("Loading dataset...")

        self.movies, self.ratings = load_and_preprocess_data()

        print("\nTraining Content-Based Model...")

        self.content_model.fit(self.movies)

        print("\nTraining Collaborative Filtering Model...")

        self.collaborative_model.fit(
            self.ratings,
            self.movies
        )

        print("\nAll recommendation models trained successfully!")

        return self

    def search_movies(self, query):

        """Search movies by title."""

        if self.movies is None:
            return []

        results = self.movies[
            self.movies["title"].str.contains(
                query,
                case=False,
                na=False,
                regex=False
            )
        ]

        return results[
            ["movieId", "title", "genres"]
        ].head(20).to_dict("records")

    def recommend_content(self, movie_title, n=10):

        return self.content_model.recommend(
            movie_title,
            n
        )

    def recommend_collaborative(self, movie_id, n=10):

        return self.collaborative_model.recommend(
            movie_id,
            n
        )

    def recommend_hybrid(self, movie_title, n=10):

        """Combine content and collaborative recommendations."""

        if self.movies is None:
            return []

        selected = self.movies[
            self.movies["title"] == movie_title
        ]

        if selected.empty:
            return []

        movie_id = int(selected.iloc[0]["movieId"])

        content_results = self.recommend_content(
            movie_title,
            n=30
        )

        collaborative_results = self.recommend_collaborative(
            movie_id,
            n=30
        )

        combined = {}

        # Add content-based recommendations
        for movie in content_results:

            movie_id_result = movie["movieId"]

            combined[movie_id_result] = {
                "movieId": movie_id_result,
                "title": movie["title"],
                "genres": movie["genres"],
                "content_score": movie["similarity_score"],
                "collaborative_score": 0.0
            }

        # Add collaborative recommendations
        for movie in collaborative_results:

            movie_id_result = movie["movieId"]

            if movie_id_result not in combined:

                combined[movie_id_result] = {
                    "movieId": movie_id_result,
                    "title": movie["title"],
                    "genres": movie["genres"],
                    "content_score": 0.0,
                    "collaborative_score": 0.0
                }

            combined[movie_id_result]["collaborative_score"] = (
                movie["similarity_score"]
            )

        results = []

        for movie in combined.values():

            # Weighted hybrid score
            hybrid_score = (
                0.5 * movie["content_score"]
                + 0.5 * movie["collaborative_score"]
            )

            movie["hybrid_score"] = round(
                hybrid_score, 4
            )

            results.append(movie)

        results.sort(
            key=lambda x: x["hybrid_score"],
            reverse=True
        )

        return results[:n]

    def get_popular_movies(self, n=10, min_ratings=20):

        """Return popular movies based on ratings."""

        if self.ratings is None:
            return []

        stats = self.ratings.groupby("movieId").agg(
            average_rating=("rating", "mean"),
            rating_count=("rating", "count")
        ).reset_index()

        stats = stats[
            stats["rating_count"] >= min_ratings
        ]

        popular = stats.merge(
            self.movies[
                ["movieId", "title", "genres"]
            ],
            on="movieId"
        )

        popular = popular.sort_values(
            ["average_rating", "rating_count"],
            ascending=[False, False]
        )

        return popular.head(n).to_dict("records")


if __name__ == "__main__":

    engine = MovieRecommendationEngine()

    engine.train()

    print("\nMOVIE SEARCH RESULTS:")

    display_movies = engine.search_movies("Toy Story")

    for movie in display_movies[:5]:
        print(movie["title"])

    print("\nHYBRID MOVIE RECOMMENDATIONS:")

    recommendations = engine.recommend_hybrid(
        "Toy Story (1995)",
        n=10
    )

    for movie in recommendations:

        print(
            movie["title"],
            "| Hybrid Score:",
            movie["hybrid_score"]
        )

    print("\nPOPULAR MOVIES:")

    popular = engine.get_popular_movies(n=10)

    for movie in popular:
        print(
            movie["title"],
            "| Rating:",
            round(movie["average_rating"], 2)
        )
