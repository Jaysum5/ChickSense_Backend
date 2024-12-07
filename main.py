from huggingface_hub import hf_hub_download
from fastapi import FastAPI, UploadFile
import io
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the pre-trained model
model_file = hf_hub_download(repo_id="Jaysum/efficientnet_poultry_disease_model.keras", filename="efficientnet_poultry_disease_model.keras")

# Load the model
model = tf.keras.models.load_model(model_file)

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
    #return {"class": CLASSES[predicted_class], "confidence": confidence}
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