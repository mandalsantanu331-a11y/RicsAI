from io import BytesIO

from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image, UnidentifiedImageError

from app.api.iot import router as iot_router
from app.predictor import predict_image
from app.schemas.prediction import PredictionResponse


app = FastAPI(
    title="Rice Disease Detection API",
    description="AI + IoT rice disease detection and field monitoring API",
    version="1.1.0",
)

app.include_router(iot_router)


@app.get("/")
def root():
    return {
        "success": True,
        "message": "Rice Disease Detection API is running",
        "features": ["AI disease prediction", "IoT field monitoring"],
    }


@app.get("/health")
def health_check():
    return {
        "success": True,
        "status": "healthy",
    }


@app.post(
    "/predict",
    response_model=PredictionResponse
)
async def predict(file: UploadFile = File(...)):

    if not file.content_type:
        raise HTTPException(
            status_code=400,
            detail="File type is missing."
        )

    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Only image files are allowed."
        )

    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents))
        result = predict_image(image)

        return {
            "success": True,
            "data": {
                "filename": file.filename,
                "prediction": result["prediction"],
                "confidence": result["confidence"],
                "probabilities": result["probabilities"],
            }
        }

    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail="Invalid or corrupted image."
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
