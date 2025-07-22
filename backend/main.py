from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from utils import read_image_from_bytes, run_inference

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class DetectionResponse(BaseModel):
    processed_image: str
    detections: list

@app.post("/predict", response_model=DetectionResponse)
async def predict(file: UploadFile = File(...)):
    if not file.filename.endswith((".jpg", ".jpeg", ".png")):
        raise HTTPException(status_code=400, detail="Invalid file type")

    try:
        image_bytes = await file.read()
        image_np = read_image_from_bytes(image_bytes)
        encoded_image, detection_info = run_inference(image_np)

        return DetectionResponse(processed_image=encoded_image, detections=detection_info)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
