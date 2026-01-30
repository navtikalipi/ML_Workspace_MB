import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------
# Load & prepare data
# -------------------------
df = pd.read_csv("spotify.csv")

FEATURES = [
    "danceability",
    "energy",
    "valence",
    "tempo",
    "duration_ms",
    "popularity"
]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[FEATURES])

# Precompute similarity matrix
SIM_MATRIX = cosine_similarity(X_scaled)

# -------------------------
# FastAPI app
# -------------------------
app = FastAPI(
    title="Spotify Recommendation API",
    version="1.0"
)

class RecommendRequest(BaseModel):
    track_name: str
    top_n: int = 5

@app.get("/health")
def health():
    return {"status": "OK"}

@app.post("/recommend")
def recommend_songs(req: RecommendRequest):
    matches = df.index[df["track_name"].str.lower() == req.track_name.lower()].tolist()

    if not matches:
        raise HTTPException(status_code=404, detail="Track not found")

    idx = matches[0]
    scores = SIM_MATRIX[idx]

    similar_idx = scores.argsort()[::-1][1:req.top_n + 1]

    result = df.loc[
        similar_idx,
        ["track_name", "artist", "genre", "playlist_category"]
    ].to_dict(orient="records")

    return {
        "input_track": req.track_name,
        "recommendations": result
    }
