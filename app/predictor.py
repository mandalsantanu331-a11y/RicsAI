import numpy as np
from PIL import Image

from app.model import model


CLASS_NAMES = [
    "bacterial_leaf_blight",
    "bacterial_leaf_streak",
    "bacterial_panicle_blight",
    "blast",
    "brown_spot",
    "dead_heart",
    "downy_mildew",
    "hispa",
    "normal",
    "tungro",
]

IMAGE_SIZE = (224, 224)


def predict_image(image: Image.Image):

    # Convert image to RGB
    image = image.convert("RGB")

    # Resize to model input size
    image = image.resize(IMAGE_SIZE)

    # Convert image to NumPy array
    # Keep pixel values in 0-255 range.
    # The model already contains Rescaling(1/255).
    image_array = np.array(
        image,
        dtype=np.float32
    )

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # Model prediction
    predictions = model.predict(
        image_array,
        verbose=0
    )[0]

    # Get predicted class
    predicted_index = int(
        np.argmax(predictions)
    )

    prediction = CLASS_NAMES[predicted_index]

    # Convert confidence from 0-1 to percentage
    confidence = float(
        predictions[predicted_index] * 100
    )

    # Convert all probabilities to percentages
    probabilities = {
        CLASS_NAMES[i]: float(
            predictions[i] * 100
        )
        for i in range(len(CLASS_NAMES))
    }

    return {
        "prediction": prediction,
        "confidence": confidence,
        "probabilities": probabilities,
    }