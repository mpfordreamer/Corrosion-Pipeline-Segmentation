import os
import io
import numpy as np
import cv2
import shutil
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
import pyodbc

app = FastAPI()

# Load models
asset_model_path = "model/DeepLabV3_Asset.keras"
corrosion_model_path = "model/DeepLabV3_Corrosion.keras"

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

    return StreamingResponse(asset_stream, media_type="image/jpeg")

@app.get("/segment/corrosion/")
def segment_corrosion():
    if uploaded_image is None:
        raise HTTPException(status_code=400, detail="No image uploaded.")
    
    corrosion_prediction = predict_image(uploaded_image, corrosion_model, corrosion_colormap)
    overlay_corrosion = overlay_segmentation(uploaded_image, corrosion_prediction)
    overlay_corrosion_bgr = cv2.cvtColor(overlay_corrosion, cv2.COLOR_RGB2BGR)

    _, img_corrosion = cv2.imencode('.jpg', overlay_corrosion_bgr)
    corrosion_stream = BytesIO(img_corrosion.tobytes())

    return StreamingResponse(corrosion_stream, media_type="image/jpeg")

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

    return StreamingResponse(intersection_stream, media_type="image/jpeg")

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

# Serve static files
app.mount("/save_images", StaticFiles(directory=SAVE_DIR), name="save_images")

@app.post("/download_images/")
async def download_images(prediction_type: PredictionType):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    saved_files = []

    # Determine the predicted images based on the given type
    if prediction_type == PredictionType.all:
        prediction_types = [PredictionType.intersection, PredictionType.asset, PredictionType.corrosion]
    else:
        prediction_types = [prediction_type]

    for pred_type in prediction_types:
        if pred_type == PredictionType.intersection:
            prediction_image = segment_intersection_function()  # This should return bytes
        elif pred_type == PredictionType.asset:
            prediction_image = segment_asset_function()  # This should return bytes
        elif pred_type == PredictionType.corrosion:
            prediction_image = segment_corrosion_function()  # This should return bytes
        else:
            raise HTTPException(status_code=400, detail="Invalid prediction type.")

        # Create a filename for the predicted image
        filename = f"{pred_type.value}_{timestamp}.jpg"
        file_path = os.path.join(SAVE_DIR, filename)

        # Save the predicted image to the folder
        with open(file_path, "wb") as f:
            f.write(prediction_image)

        # Store the URL for downloading
        saved_files.append(f"/save_images/{filename}")  # URL to access the saved file

    # Return the URLs for the files
    return {"files": saved_files}

# Run the application if this file is executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 