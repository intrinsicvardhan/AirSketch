import tensorflow as tf
from keras_ocr import pipeline as keras_ocr_pipeline

# Check if CUDA is available
if tf.test.is_gpu_available():
    # Enable GPU memory growth to allocate memory as needed
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for gpu in physical_devices:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("CUDA is available. GPU acceleration enabled.")
else:
    print("CUDA is not available. Using CPU.")

# Load the keras-ocr pipeline
pipeline = keras_ocr_pipeline.Pipeline()

def predict_character(img):
    # Use the keras-ocr pipeline to recognize text in the image
    prediction_groups = pipeline.recognize([img])

    # Extract the recognized words from the predictions
    predictions = prediction_groups[0]

    # Extract the full recognized text
    recognized_text = ' '.join([word[0] for word in predictions])

    return recognized_text

# Example usage:
# img = keras_ocr.tools.read('path_to_your_image.jpg')
# recognized_text = predict_character(img)
# print(recognized_text)
