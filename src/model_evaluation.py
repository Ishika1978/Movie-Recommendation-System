
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.data_preprocessing import load_and_preprocess_data
from src.collaborative_filtering import CollaborativeFiltering


class ModelEvaluator:

    def __init__(self):
        self.movies = None
        self.ratings = None
        self.train_data = None
        self.test_data = None

    def load_data(self):

        self.movies, self.ratings = load_and_preprocess_data()

        self.train_data, self.test_data = train_test_split(
            self.ratings,
            test_size=0.2,
            random_state=42
        )

        print("Training records:", len(self.train_data))
        print("Testing records:", len(self.test_data))

    def evaluate_rating_prediction(self):

        """Evaluate a simple user-average rating baseline."""

        user_means = self.train_data.groupby(
            "userId"
        )["rating"].mean()

        global_mean = self.train_data["rating"].mean()

        predictions = (
            self.test_data["userId"]
            .map(user_means)
            .fillna(global_mean)
        )

        actual = self.test_data["rating"]

        mae = mean_absolute_error(
            actual,
            predictions
        )

        rmse = np.sqrt(
            mean_squared_error(
                actual,
                predictions
            )
        )

        print("\nRating Prediction Evaluation")
        print("----------------------------")
        print("MAE:", round(mae, 4))
        print("RMSE:", round(rmse, 4))

        return {
            "MAE": mae,
            "RMSE": rmse
        }

    def evaluate_recommendations(self, k=10, max_users=100):

        """Evaluate item-based KNN recommendations."""

        print("\nTraining KNN on training data...")

        recommender = CollaborativeFiltering()

        recommender.fit(
            self.train_data,
            self.movies
        )

        # Relevant movies have ratings >= 4
        relevant_items = (
            self.test_data[
                self.test_data["rating"] >= 4
            ]
            .groupby("userId")["movieId"]
            .apply(set)
            .to_dict()
        )

        train_user_movies = (
            self.train_data
            .groupby("userId")["movieId"]
            .apply(set)
            .to_dict()
        )

        precisions = []
        recalls = []
        hits = 0
        evaluated = 0

        users = list(relevant_items.keys())[:max_users]

        for user_id in users:

            relevant = relevant_items[user_id]
            watched = train_user_movies.get(user_id, set())

            if not relevant or not watched:
                continue

            # Select a movie the user rated highly in training
            seed_movie = next(iter(watched))

            recommendations = recommender.recommend(
                seed_movie,
                n=k
            )

            recommended_ids = {
                item["movieId"]
                for item in recommendations
            }

            # Never recommend movies already watched
            recommended_ids -= watched

            # Calculate ranking metrics
            true_positives = len(
                recommended_ids.intersection(relevant)
            )

            precision = true_positives / k
            recall = true_positives / len(relevant)

            precisions.append(precision)
            recalls.append(recall)

            hits += true_positives
            evaluated += 1

        if evaluated == 0:
            print("Not enough users to evaluate.")
            return None

        precision_at_k = np.mean(precisions)
        recall_at_k = np.mean(recalls)

        print("\nRecommendation Evaluation")
        print("-------------------------")
        print("Users evaluated:", evaluated)
        print(f"Precision@{k}:", round(precision_at_k, 4))
        print(f"Recall@{k}:", round(recall_at_k, 4))

        return {
            "Precision@K": precision_at_k,
            "Recall@K": recall_at_k,
            "Users Evaluated": evaluated
        }

    def run_evaluation(self):

        print("\n================================")
        print("MOVIE RECOMMENDATION EVALUATION")
        print("================================")

        self.load_data()

        rating_results = self.evaluate_rating_prediction()

        recommendation_results = self.evaluate_recommendations(
            k=10,
            max_users=100
        )

        print("\n================================")
        print("EVALUATION COMPLETED")
        print("================================")

        return rating_results, recommendation_results


if __name__ == "__main__":

    evaluator = ModelEvaluator()

    evaluator.run_evaluation()
