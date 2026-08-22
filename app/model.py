import tensorflow as tf

MODEL_PATH = "models/E1_baseline_best.keras"

print("Loading rice disease model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("Model loaded successfully!")
print("Input shape:", model.input_shape)
print("Output shape:", model.output_shape)