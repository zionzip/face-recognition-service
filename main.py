from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import numpy as np
import cv2

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/embed")
async def embed(file: UploadFile = File(...)):
    from deepface import DeepFace
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image")
    try:
        result = DeepFace.represent(img, model_name="Facenet", enforce_detection=False)
        if not result or "embedding" not in result[0]:
            raise HTTPException(status_code=422, detail="Could not extract embedding")
        return {"embedding": result[0]["embedding"]}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"No face detected: {str(e)}")

class VerifyRequest(BaseModel):
    embedding1: list[float]
    embedding2: list[float]

@app.post("/verify")
def verify(req: VerifyRequest):
    e1 = np.array(req.embedding1)
    e2 = np.array(req.embedding2)
    similarity = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))
    return {"verified": bool(similarity >= 0.4), "similarity": float(similarity)}
