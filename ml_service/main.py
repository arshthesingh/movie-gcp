from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from robust_movie_recommender import MovieRecommender

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "combiniedmovies_2.csv")
EMBEDDINGS_PATH = os.path.join(BASE_DIR, "overview_embeddings.npy")

app = FastAPI(title="Movie Recommendation ML Service")

recommender = None  # Global placeholder

@app.on_event("startup")
async def startup_event():
    global recommender
    # Initialize the recommender here after the server has started.
    recommender = MovieRecommender(CSV_PATH, EMBEDDINGS_PATH, device="cpu", use_faiss=True)

class RecommendationRequest(BaseModel):
    title: str
    top_n: int = 5
    min_vote: float = 0.0
    plot_weight: float = 0.6
    genre_weight: float = 0.3
    sentiment_weight: float = 0.1

@app.post("/recommend", summary="Get movie recommendations")
def get_recommendations(request: RecommendationRequest):
    if recommender is None:
        raise HTTPException(status_code=503, detail="Service is starting up, try again later")
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
    
    return {"recommendations": [{"title": title, "score": score} for title, score in recs]}
