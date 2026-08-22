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

    # Convert to RGB
    image = image.convert("RGB")

    # Resize exactly like training
    image = image.resize(IMAGE_SIZE)

    # Convert image to NumPy array
    # Keep pixel range 0-255 because the model
    # already contains Rescaling(1.0 / 255)
    image_array = np.array(image, dtype=np.float32)

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # Predict
    predictions = model.predict(
        image_array,
        verbose=0
    )[0]

    # Get predicted class
    predicted_index = int(np.argmax(predictions))

    disease = CLASS_NAMES[predicted_index]

    confidence = float(predictions[predicted_index])

    probabilities = {
        CLASS_NAMES[i]: float(predictions[i])
        for i in range(len(CLASS_NAMES))
    }

    return {
        "disease": disease,
        "confidence": confidence,
        "probabilities": probabilities,
    }