from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
from io import BytesIO

from app.predictor import predict_image


app = FastAPI(
    title="Rice Disease Detection API",
    description="CNN-based rice leaf disease classification API",
    version="1.0.0",
)


@app.get("/")
def root():
    return {
        "success": True,
        "message": "Rice Disease Detection API is running",
    }


@app.get("/health")
def health_check():
    return {
        "success": True,
        "status": "healthy",
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Validate content type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image file.",
        )

    try:
        # Read uploaded file
        contents = await file.read()

        # Open image
        image = Image.open(BytesIO(contents))

        # Predict
        result = predict_image(image)

        return {
            "success": True,
            "filename": file.filename,
            "prediction": result["disease"],
            "confidence": result["confidence"],
            "probabilities": result["probabilities"],
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}",
        )