from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from robust_movie_recommender import MovieRecommender

# Set up paths for your CSV and embeddings file.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "combiniedmovies_2.csv")  # adjust the filename if needed
EMBEDDINGS_PATH = os.path.join(BASE_DIR, "overview_embeddings.npy")

app = FastAPI(title="Movie Recommendation ML Service")

# Instantiate the recommender as a global object.
# (If you use lazy-loading inside the recommender, this won’t load heavy resources until needed.)
recommender = MovieRecommender(CSV_PATH, EMBEDDINGS_PATH, device="cpu", use_faiss=True)

# Define the request model using Pydantic.
class RecommendationRequest(BaseModel):
    title: str
    top_n: int = 5
    min_vote: float = 0.0
    plot_weight: float = 0.6
    genre_weight: float = 0.3
    sentiment_weight: float = 0.1

@app.post("/recommend", summary="Get movie recommendations")
def get_recommendations(request: RecommendationRequest):
    try:
        recs = recommender.recommend(
            movie_title=request.title,
            top_n=request.top_n,
            min_vote=request.min_vote,
            plot_weight=request.plot_weight,
            genre_weight=request.genre_weight,
            sentiment_weight=request.sentiment_weight,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    if not recs:
        raise HTTPException(status_code=404, detail=f"No recommendations found for '{request.title}'")
    
    # Return a list of dictionaries with movie titles and their scores.
    return {"recommendations": [{"title": title, "score": score} for title, score in recs]}

# To run this service, use:
#   uvicorn main:app --host 0.0.0.0 --port 8000
