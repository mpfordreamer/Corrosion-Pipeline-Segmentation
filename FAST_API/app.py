import os
import io
import numpy as np
import cv2
import math
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from tensorflow.keras.models import load_model
from PIL import Image
import uuid
import datetime
from typing import List
from fastapi.responses import FileResponse
from io import BytesIO
from enum import Enum
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Header, HTTPException
from typing import List, Optional 
from fastapi.staticfiles import StaticFiles
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import logging

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)

# Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5001"],  # Allow only requests from this origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
asset_model_path = "model/DeepLabV3Plus_256A_B16E100_Asset.keras"
corrosion_model_path = "model/DeepLabV3Plus_256A_B16E100_Corrosion.keras"

# Ensure the model paths are correct before loading
if not os.path.exists(asset_model_path):
    raise FileNotFoundError(f"Asset model file not found at {asset_model_path}")
if not os.path.exists(corrosion_model_path):
    raise FileNotFoundError(f"Corrosion model file not found at {corrosion_model_path}")

asset_model = load_model(asset_model_path)
corrosion_model = load_model(corrosion_model_path)

# Folder untuk menyimpan gambar hasil prediksi
SAVE_DIR = "save_images"
os.makedirs(SAVE_DIR, exist_ok=True)

# Enum untuk jenis prediksi
class PredictionType(str, Enum):
    intersection = "intersection"
    asset = "asset"
    corrosion = "corrosion"
    all = "all"

# Define colormap for visualization
asset_colormap = {
    0: [0, 0, 0],    # Black for background
    1: [0, 0, 255]   # Blue for asset
}

corrosion_colormap = {
    0: [0, 0, 0],    # Black for background
    1: [255, 0, 0]   # Red for corrosion
}


# Fungsi untuk menyimpan data ke tabel images
def save_image(filename, prediction_type):
    connection = get_db_connection()
    cursor = connection.cursor()
    query = "INSERT INTO images (filename, prediction_type) VALUES (%s, %s)"
    cursor.execute(query, (filename, prediction_type))
    connection.commit()
    cursor.close()
    connection.close()

# Fungsi untuk menyimpan data ke tabel segment_data
def save_segment_data(image_id, asset_pixel, corrosion_pixel, corrosion_in_asset_pixel, asset_dominance, corrosion_dominance, corrosion_in_asset_dominance):
    connection = get_db_connection()
    cursor = connection.cursor()
    query = "INSERT INTO segment_data (image_id, asset_pixel, corrosion_pixel, corrosion_in_asset_pixel, asset_dominance, corrosion_dominance, corrosion_in_asset_dominance) VALUES (%s, %s, %s, %s, %s, %s, %s)"
    cursor.execute(query, (image_id, asset_pixel, corrosion_pixel, corrosion_in_asset_pixel, asset_dominance, corrosion_dominance, corrosion_in_asset_dominance))
    connection.commit()
    cursor.close()
    connection.close()


# Preprocessing and utility functions
def predict_image(image, model, colormap):
    """Predict image using the given model and apply specific class colors."""
    input_image = cv2.resize(image, (256, 256))  # Resize image to match model input size
    input_image = np.expand_dims(input_image, axis=0)  # Add batch dimension

    prediction = model.predict(input_image)  # Get model prediction
    predicted_class = np.argmax(prediction[0], axis=-1)  # Get the class index with max probability

    # Create an RGB prediction map
    rgb_prediction = np.zeros((256, 256, 3), dtype=np.uint8)
    for class_index, color in colormap.items():
        rgb_prediction[predicted_class == class_index] = color

    return rgb_prediction

def intersection_segmentations(asset_segmentation, corrosion_segmentation):
    """intersection asset and corrosion segmentations into a single image, only showing corrosion within asset."""
    asset_mask = np.all(asset_segmentation == [0, 0, 255], axis=-1)  # Asset pixels (Blue)
    corrosion_mask = np.all(corrosion_segmentation == [255, 0, 0], axis=-1)  # Corrosion pixels (Red)

    # Initialize intersection segmentation
    intersection_segmentation = np.zeros_like(asset_segmentation)

    # Apply masks
    intersection_segmentation[asset_mask] = [0, 0, 255]  # Blue for asset
    intersection_segmentation[np.logical_and(corrosion_mask, asset_mask)] = [255, 0, 0]  # Red for corrosion within asset

    return intersection_segmentation

def overlay_segmentation(original_image, segmentation, alpha=0.5):
    """Overlay segmentation onto the original image."""
    # Resize segmentation to match the original image size
    segmentation_resized = cv2.resize(segmentation, (original_image.shape[1], original_image.shape[0]))
    
    # Perform overlay
    return cv2.addWeighted(original_image, 1 - alpha, segmentation_resized, alpha, 0)

# Global variable to hold the uploaded image
uploaded_image = None


# Endpoints
@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    global uploaded_image
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Invalid file type. Only PNG, JPG, and JPEG are supported.")
    
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    
    if image is None:
        raise HTTPException(status_code=400, detail="Error loading image.")
    
    uploaded_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return {"message": "Image uploaded successfully."}

def segment_asset_function():
    if uploaded_image is None:
        raise HTTPException(status_code=400, detail="No image uploaded.")
    
    asset_prediction = predict_image(uploaded_image, asset_model, asset_colormap)
    overlay_asset = overlay_segmentation(uploaded_image, asset_prediction)
    overlay_asset_bgr = cv2.cvtColor(overlay_asset, cv2.COLOR_RGB2BGR)

    _, img_asset = cv2.imencode('.jpg', overlay_asset_bgr)
    return img_asset.tobytes()  # Return image as bytes

def segment_corrosion_function():
    if uploaded_image is None:
        raise HTTPException(status_code=400, detail="No image uploaded.")
    
    corrosion_prediction = predict_image(uploaded_image, corrosion_model, corrosion_colormap)
    overlay_corrosion = overlay_segmentation(uploaded_image, corrosion_prediction)
    overlay_corrosion_bgr = cv2.cvtColor(overlay_corrosion, cv2.COLOR_RGB2BGR)

    _, img_corrosion = cv2.imencode('.jpg', overlay_corrosion_bgr)
    return img_corrosion.tobytes()  # Return image as bytes

def segment_intersection_function():
    if uploaded_image is None:
        raise HTTPException(status_code=400, detail="No image uploaded.")
    
    asset_prediction = predict_image(uploaded_image, asset_model, asset_colormap)
    corrosion_prediction = predict_image(uploaded_image, corrosion_model, corrosion_colormap)
    intersection_segmentation = intersection_segmentations(asset_prediction, corrosion_prediction)
    overlay_intersection = overlay_segmentation(uploaded_image, intersection_segmentation)
    overlay_intersection_bgr = cv2.cvtColor(overlay_intersection, cv2.COLOR_RGB2BGR)

    _, img_intersection = cv2.imencode('.jpg', overlay_intersection_bgr)
    return img_intersection.tobytes()  # Return image as bytes

@app.get("/segment/asset/")
def segment_asset():
    if uploaded_image is None:
        raise HTTPException(status_code=400, detail="No image uploaded.")
    
    asset_prediction = predict_image(uploaded_image, asset_model, asset_colormap)
    overlay_asset = overlay_segmentation(uploaded_image, asset_prediction)
    overlay_asset_bgr = cv2.cvtColor(overlay_asset, cv2.COLOR_RGB2BGR)

    _, img_asset = cv2.imencode('.jpg', overlay_asset_bgr)
    asset_stream = BytesIO(img_asset.tobytes())

    # Calculate total pixels and asset pixels
    total_pixels = uploaded_image.shape[0] * uploaded_image.shape[1]
    asset_mask = np.all(asset_prediction == [0, 0, 255], axis=-1)  # Pixels identified as asset
    asset_pixels = int(np.sum(asset_mask))

    return StreamingResponse(asset_stream, media_type="image/jpeg", headers={
        "total_pixel": str(total_pixels),
        "asset_pixel": str(asset_pixels)
    })

@app.get("/segment/corrosion/")
def segment_corrosion():
    if uploaded_image is None:
        raise HTTPException(status_code=400, detail="No image uploaded.")
    
    corrosion_prediction = predict_image(uploaded_image, corrosion_model, corrosion_colormap)
    overlay_corrosion = overlay_segmentation(uploaded_image, corrosion_prediction)
    overlay_corrosion_bgr = cv2.cvtColor(overlay_corrosion, cv2.COLOR_RGB2BGR)

    _, img_corrosion = cv2.imencode('.jpg', overlay_corrosion_bgr)
    corrosion_stream = BytesIO(img_corrosion.tobytes())

    # Calculate total pixels and corrosion pixels
    total_pixels = uploaded_image.shape[0] * uploaded_image.shape[1]
    corrosion_mask = np.all(corrosion_prediction == [255, 0, 0], axis=-1)  # Pixels identified as corrosion
    corrosion_pixels = int(np.sum(corrosion_mask))

    return StreamingResponse(corrosion_stream, media_type="image/jpeg", headers={
        "total_pixel": str(total_pixels),
        "corrosion_pixels": str(corrosion_pixels)
    })

@app.get("/segment/intersection/")
def segment_intersection():
    if uploaded_image is None:
        raise HTTPException(status_code=400, detail="No image uploaded.")
    
    asset_prediction = predict_image(uploaded_image, asset_model, asset_colormap)
    corrosion_prediction = predict_image(uploaded_image, corrosion_model, corrosion_colormap)
    intersection_segmentation = intersection_segmentations(asset_prediction, corrosion_prediction)
    overlay_intersection = overlay_segmentation(uploaded_image, intersection_segmentation)
    overlay_intersection_bgr = cv2.cvtColor(overlay_intersection, cv2.COLOR_RGB2BGR)

    _, img_intersection = cv2.imencode('.jpg', overlay_intersection_bgr)
    intersection_stream = BytesIO(img_intersection.tobytes())

    # Calculate total pixels, asset pixels, and corrosion pixels
    total_pixels = uploaded_image.shape[0] * uploaded_image.shape[1]
    asset_mask = np.all(asset_prediction == [0, 0, 255], axis=-1)  # Pixels identified as asset
    corrosion_mask = np.all(corrosion_prediction == [255, 0, 0], axis=-1)  # Pixels identified as corrosion
    asset_pixels = int(np.sum(asset_mask))
    corrosion_pixels = int(np.sum(corrosion_mask))

    # Calculate percentages
    asset_data = (asset_pixels / total_pixels) * 100
    corrosion_in_asset_pixels = int(np.sum(np.logical_and(asset_mask, corrosion_mask)))
    corrosion_in_asset_data = (corrosion_in_asset_pixels / asset_pixels) * 100 if asset_pixels > 0 else 0

    return StreamingResponse(intersection_stream, media_type="image/jpeg", headers={
        "total_pixel": str(total_pixels),
        "asset_pixel": str(asset_pixels),
        "corrosion_pixels": str(corrosion_pixels),
        "asset_of_Image": f"{round(asset_data, 2)}%",
        "corrosion_of_asset": f"{round(corrosion_in_asset_data, 2)}%"
    })

@app.get("/segment/all/")
def segment_all():
    if uploaded_image is None:
        raise HTTPException(status_code=400, detail="No image uploaded.")
    
    # Predict asset, corrosion, and intersection segmentations
    asset_prediction = predict_image(uploaded_image, asset_model, asset_colormap)
    corrosion_prediction = predict_image(uploaded_image, corrosion_model, corrosion_colormap)
    intersection_segmentation = intersection_segmentations(asset_prediction, corrosion_prediction)
    
    # Overlay segmentations
    overlay_asset = overlay_segmentation(uploaded_image, asset_prediction)
    overlay_corrosion = overlay_segmentation(uploaded_image, corrosion_prediction)
    overlay_intersection = overlay_segmentation(uploaded_image, intersection_segmentation)

    # Resize images to make them the same size for display
    original_resized = cv2.resize(uploaded_image, (256, 256))
    overlay_asset_resized = cv2.resize(overlay_asset, (256, 256))
    overlay_corrosion_resized = cv2.resize(overlay_corrosion, (256, 256))
    overlay_intersection_resized = cv2.resize(overlay_intersection, (256, 256))

    # Stack images horizontally for side-by-side display
    stacked_image = np.hstack([
        original_resized,
        overlay_asset_resized,
        overlay_corrosion_resized,
        overlay_intersection_resized
    ])
    
    # Convert to BGR for response
    stacked_image_bgr = cv2.cvtColor(stacked_image, cv2.COLOR_RGB2BGR)

    # Encode image and send it as a stream
    _, img_all = cv2.imencode('.jpg', stacked_image_bgr)
    all_stream = BytesIO(img_all.tobytes())

    return StreamingResponse(all_stream, media_type="image/jpeg")

    
@app.get("/segment/data/")
def segment_data():
    """
    Hitung dominasi kelas aset, korosi, dan korosi dalam aset berdasarkan jumlah piksel dari gambar asli.
    """
    if uploaded_image is None:
        raise HTTPException(status_code=400, detail="No image uploaded.")
    
    # Predict asset and corrosion segmentations
    asset_prediction = predict_image(uploaded_image, asset_model, asset_colormap)
    corrosion_prediction = predict_image(uploaded_image, corrosion_model, corrosion_colormap)
    
    # Resize predictions to match the original image size
    asset_prediction_resized = cv2.resize(asset_prediction, (uploaded_image.shape[1], uploaded_image.shape[0]))
    corrosion_prediction_resized = cv2.resize(corrosion_prediction, (uploaded_image.shape[1], uploaded_image.shape[0]))
    
    # Calculate masks
    asset_mask = np.all(asset_prediction_resized == [0, 0, 255], axis=-1)  # Pixels identified as asset
    corrosion_mask = np.all(corrosion_prediction_resized == [255, 0, 0], axis=-1)  # Pixels identified as corrosion
    corrosion_in_asset_mask = np.logical_and(asset_mask, corrosion_mask)  # Corrosion pixels within asset

    # Total pixels in the image
    total_pixels = uploaded_image.shape[0] * uploaded_image.shape[1]
    
    # Count pixels
    asset_pixels = int(np.sum(asset_mask))
    corrosion_pixels = int(np.sum(corrosion_mask))
    corrosion_in_asset_pixels = int(np.sum(corrosion_in_asset_mask))
    
    # Calculate percentages
    asset_data = (asset_pixels / total_pixels) * 100
    corrosion_data = (corrosion_pixels / total_pixels) * 100
    corrosion_in_asset_data = (corrosion_in_asset_pixels / asset_pixels) * 100

    return {
        "total_pixel": total_pixels,
        "asset_pixel": asset_pixels,
        "corrosion_pixels": corrosion_pixels,
        "asset_of_Image": f"{round(asset_data, 2)}%",
        "corrosion_of_asset": f"{round(corrosion_in_asset_data, 2)}%"
    }

# Function to validate token (currently does nothing)
def validate_token(token: str):
    # This function can be kept for future use or removed entirely
    # Currently, it does nothing and is not called anywhere
    pass

# Function to check if authentication should be ignored
def should_ignore_auth():
    # Implement your logic here. For example, return True to ignore auth.
    return True  # Change this logic as needed

# # Serve static files
# app.mount("/save_images", StaticFiles(directory=SAVE_DIR), name="save_images")

@app.post("/save_images/")
async def save_images(prediction_type: PredictionType):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    saved_files = []

    # Tentukan gambar yang diprediksi berdasarkan tipe yang diberikan
    if prediction_type == PredictionType.all:
        prediction_types = [PredictionType.intersection, PredictionType.asset, PredictionType.corrosion]
    else:
        prediction_types = [prediction_type]

    for pred_type in prediction_types:
        if pred_type == PredictionType.intersection:
            prediction_image = segment_intersection_function()  # Ini harus mengembalikan bytes
        elif pred_type == PredictionType.asset:
            prediction_image = segment_asset_function()  # Ini harus mengembalikan bytes
        elif pred_type == PredictionType.corrosion:
            prediction_image = segment_corrosion_function()  # Ini harus mengembalikan bytes
        else:
            raise HTTPException(status_code=400, detail="Invalid prediction type.")

        # Buat nama file untuk gambar yang diprediksi
        filename = f"{pred_type.value}_{timestamp}.jpg"
        file_path = os.path.join(SAVE_DIR, filename)

        # Simpan gambar yang diprediksi ke folder
        with open(file_path, "wb") as f:
            f.write(prediction_image)

        # Simpan URL untuk mengunduh
        saved_files.append(f"/save_images/{filename}")  # URL untuk mengakses file yang disimpan

    # Kembalikan URL untuk file
    return {"files": saved_files}

@app.options("/save_images/")
async def options_save_images():
    return JSONResponse(
        content={},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
        },
    )
# Run the application if this file is executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
   