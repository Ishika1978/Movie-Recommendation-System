
from flask import Flask, render_template, request, jsonify

from src.recommendation_engine import MovieRecommendationEngine

app = Flask(__name__)

# Initialize recommendation engine
engine = None


def get_engine():
    global engine

    if engine is None:
        print("Loading Movie Recommendation System...")

        engine = MovieRecommendationEngine()
        engine.train()

        print("Recommendation System Ready!")

    return engine


# =====================================
# HOME PAGE
# =====================================

@app.route("/")
def home():

    model = get_engine()

    return render_template(
        "index.html",
        movie_count=len(model.movies),
        rating_count=len(model.ratings),
        user_count=model.ratings["userId"].nunique()
    )


# =====================================
# SEARCH MOVIES API
# =====================================

@app.route("/api/search")
def search_movies():

    model = get_engine()

    query = request.args.get("q", "").strip()

    if not query:
        return jsonify([])

    results = model.search_movies(query)

    return jsonify(results)


# =====================================
# RECOMMENDATION API
# =====================================

@app.route("/api/recommend", methods=["POST"])
def recommend_movies():

    model = get_engine()

    data = request.get_json(silent=True) or {}

    movie_id = data.get("movie_id")
    algorithm = data.get("algorithm", "hybrid")

    try:
        movie_id = int(movie_id)
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid movie ID"}), 400

    selected = model.movies[
        model.movies["movieId"] == movie_id
    ]

    if selected.empty:
        return jsonify({"error": "Movie not found"}), 404

    movie_title = selected.iloc[0]["title"]

    if algorithm == "content":

        results = model.recommend_content(
            movie_title, n=10
        )

    elif algorithm == "collaborative":

        results = model.recommend_collaborative(
            movie_id, n=10
        )

    elif algorithm == "hybrid":

        results = model.recommend_hybrid(
            movie_title, n=10
        )

    else:
        return jsonify({"error": "Invalid algorithm"}), 400

    return jsonify(results)


# =====================================
# POPULAR MOVIES API
# =====================================

@app.route("/api/popular")
def popular_movies():

    model = get_engine()

    try:
        number = int(request.args.get("limit", 10))
        minimum = int(request.args.get("min_ratings", 20))

        number = max(1, min(number, 30))
        minimum = max(1, minimum)

    except ValueError:
        return jsonify({"error": "Invalid parameters"}), 400

    results = model.get_popular_movies(
        n=number,
        min_ratings=minimum
    )

    return jsonify(results)


# =====================================
# RUN FLASK APPLICATION
# =====================================

if __name__ == "__main__":

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )
