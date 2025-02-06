import os
import requests
from flask import Blueprint, jsonify, request
from flask_jwt_extended import jwt_required

recommendations_blueprint = Blueprint("recommendations", __name__)

# MODEL_SERVICE_URL should be set in your environment;
# for example, in Docker Compose you might use the service name.
MODEL_SERVICE_URL = os.getenv("MODEL_SERVICE_URL", "http://ml_service:8000/recommend")

@recommendations_blueprint.route("/recommendations", methods=["GET"])
@jwt_required()
def get_recommendations():
    title = request.args.get("title")
    if not title:
        return jsonify({"error": "Movie title is required"}), 400

    params = {
        "title": title,
        "top_n": request.args.get("top_n", default=5, type=int),
        "min_vote": request.args.get("min_vote", default=0.0, type=float),
        "plot_weight": request.args.get("plot_weight", default=0.6, type=float),
        "genre_weight": request.args.get("genre_weight", default=0.3, type=float),
        "sentiment_weight": request.args.get("sentiment_weight", default=0.1, type=float),
    }

    try:
        response = requests.post(MODEL_SERVICE_URL, json=params)
        response.raise_for_status()
    except requests.RequestException as e:
        return jsonify({"error": f"Error contacting model service: {str(e)}"}), 500

    return jsonify(response.json()), 200
