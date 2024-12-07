from fastapi import FastAPI, UploadFile
import io
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import requests

# Google Drive file ID of the model
DRIVE_FILE_ID = "1HBzc72rm8NpJZoMQwLA4SJOBR_FzMjRe"  # Replace with your file ID
MODEL_PATH = "./efficientnet_poultry_disease_model.keras"

# Function to download the model from Google Drive
def download_model():
    if not os.path.exists(MODEL_PATH):  # Check if the model already exists
        print("Downloading model from Google Drive...")
        # Construct the download URL
        url = f"https://drive.google.com/uc?export=download&id={DRIVE_FILE_ID}"
        with requests.Session() as session:
            response = session.get(url, stream=True)
            # Handle confirmation for large files
            for key, value in response.cookies.items():
                if key.startswith("download_warning"):
                    url = f"https://drive.google.com/uc?export=download&id={DRIVE_FILE_ID}&confirm={value}"
                    response = session.get(url, stream=True)
                    break
            # Write the content to a file
            with open(MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=32768):
                    f.write(chunk)

# Download the model if not already downloaded
download_model()

# Load the pre-trained model
model = tf.keras.models.load_model(MODEL_PATH)

# List of classes the model predicts
CLASSES = ['coccidiosis', 'healthy', 'newcastle disease', 'salmo']
IMAGE_SIZE = (360, 360)

# Function to preprocess and predict a single image
def predict_image(image_stream: io.BytesIO):
    # Load the image from the BytesIO stream
    img = Image.open(image_stream)
    img = img.resize(IMAGE_SIZE)  # Resize the image
    img_array = np.array(img, dtype=np.float32)  # Convert to float32 for proper scaling
    img_array = np.expand_dims(img_array, axis=0)  # Expand dims to make it (1, IMAGE_SIZE, IMAGE_SIZE, 3)

    # Normalize the image by dividing by 255.0
    img_array /= 255.0

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = predictions[0][predicted_class]

    # Return prediction results as a dictionary
    return {
        "class": CLASSES[predicted_class],
        "confidence": confidence.item()  # Convert numpy.float32 to native float
    }

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/predict")
async def predict(image: UploadFile):
    content = await image.read()  # Read the uploaded image content

    # Pass the byte content as a BytesIO object to predict_image
    result = predict_image(io.BytesIO(content))  # Pass the byte stream instead of the PIL image

    return result  # FastAPI will automatically convert the dictionary to JSON
