# **Image Segmentation Project: Class Detection in Corroded Pipe Imagery with U-Net & DeepLabV3+**
<img width="500" alt="Portfolio" src="https://github.com/user-attachments/assets/2a79f682-31f5-45d7-a32a-2bbcf148c359" />

## **Project Summary**
This project aims to build and train **U-Net** & **DeepLabV3+** deep learning models for an image segmentation task. The model is designed to classify each pixel in an image into predefined classes (*background*, *asset*, *corrosion*). By leveraging a powerful encoder-decoder architecture and *Atrous Spatial Pyramid Pooling* (ASPP), the model can capture multi-scale context to produce accurate segmentation predictions.

### **Project Objectives:**

1.  **Build a Segmentation Model**: Implement the U-Net & DeepLabV3+ architectures with a ResNet50 backbone to perform pixel-wise classification.
2.  **Achieve High Accuracy**: Train the model to achieve optimal evaluation metrics such as *Intersection over Union* (IoU) and *Dice Coefficient*.
3.  **Visualize Results**: Generate prediction masks that can visually distinguish between classes in a test image.

### **Project Scope:**

*   Preprocessing of image data and ground truth masks.
*   Implementation and training of the DeepLabV3+ model using TensorFlow and Keras.
*   Evaluation of model performance on test data using standard segmentation metrics.
*   Visualization of prediction results for qualitative analysis.

### **Data Preparation**

#### **Dataset:**

The dataset used consists of original images and their corresponding ground truth masks, organized in the following folder structure:
```
/DATASET
    /original
        ORI_001.png
        ...
    /ground_truth
        GT_001.png
        ...
```
**Dataset Details:**
-   **Total Data**: 216 pairs of images and masks.
-   **Format**: PNG
-   **Image Size**: Standardized to 256x256 pixels.
-   **Color Mode**: RGB

Ensure your environment is installed with the necessary libraries:
```bash
pip install tensorflow numpy opencv-python matplotlib seaborn openpyxl
```

#### **Data Preparation Process:**
1.  **Normalization & Standardization**: All images and masks are resized to 256x256 and normalized.
2.  **Data Splitting**: The dataset is randomly split into training (80%), validation (10%), and test (10%) sets.
3.  **One-Hot Encoding**: The RGB ground truth masks are converted into a one-hot encoded format with 3 classes to be used as the training target.
    *   **Class 0**: Red `(255, 0, 0)`
    *   **Class 1**: Blue `(0, 0, 255)`
    *   **Class 2**: Green `(0, 255, 0)`

### **Modeling**

The model built is **DeepLabV3+**, a state-of-the-art architecture for semantic segmentation.

*   **Encoder**: Uses **ResNet50**, pre-trained on ImageNet, to extract features from the input image.
*   **Atrous Spatial Pyramid Pooling (ASPP)**: This block is used to capture context at multiple scales without reducing spatial resolution, which is crucial for segmenting objects of different sizes.
*   **Decoder**: Combines features from the encoder with features from ASPP, then performs upsampling to produce a segmentation mask with the same resolution as the original image.
*   **Output Activation Function**: Uses `softmax` to generate class probabilities for each pixel.

### **Evaluation**

The model's performance is evaluated using standard metrics for segmentation tasks:

*   **Intersection over Union (IoU)**: Measures the overlap between the predicted mask and the ground truth.
*   **Dice Coefficient**: Similar to IoU, this metric also measures overlap and is very common in segmentation.
*   **F1-Score**: Provides a single score that balances precision and recall.
*   **Confusion Matrix**: Provides a visual overview of classification performance for each class.

Here is a summary of the evaluation results for both models on the test data:

**DeepLabV3+**
| Class             | Average IoU | Average Dice | Average F1-Score |
| ----------------- | ----------- | ------------ | ---------------- |
| Corrosion (Red)   | 56.66%      | 64.19%       | 88.65%           |
| Asset (Blue)      | 85.73%      | 91.55%       | 93.65%           |
| **Overall**       | **80.07%**  | **80.07%**   | **80.07%**       |

**U-Net**
| Class             | Average IoU | Average Dice | Average F1-Score |
| ----------------- | ----------- | ------------ | ---------------- |
| Corrosion (Red)   | 52.54%      | 59.00%       | 85.38%           |
| Asset (Blue)      | 70.07%      | 79.93%       | 83.63%           |
| **Overall**       | **71.76%**  | **71.76%**   | **71.76%**       |

### **Architecture & Deployment**

The trained DeepLabV3+ model is integrated into an application system for practical use through a service-based architecture (*microservices*).

*   **Backend (FastAPI)**: A backend API is built using **FastAPI** (Python) to serve the machine learning model. This endpoint is responsible for receiving an image, processing it with the DeepLabV3+ model, and returning the segmentation result.
*   **Frontend (ASP.NET)**: The user-facing web application (interface) is developed using **ASP.NET**. Users can upload images through this interface, which are then sent to the FastAPI backend for processing.

**System Workflow:**
1.  The user accesses the ASP.NET web application and uploads an image.
2.  The ASP.NET frontend sends the image to the FastAPI API endpoint.
3.  The FastAPI backend receives the image, preprocesses it, and feeds it to the DeepLabV3+ model for prediction.
4.  The model generates a segmentation mask.
5.  FastAPI returns the predicted mask to the ASP.NET application.
6.  The frontend displays the original image alongside its segmentation result to the user.

### **Conclusion**

This project successfully implemented a **DeepLabV3+** model for a multi-class segmentation task—outperforming U-Net with an overall average metric of 80.07% compared to 71.76%. The DeepLabV3+ architecture with a ResNet50 backbone and ASPP proved to be more effective at capturing pixel context to distinguish between corrosion and assets. This feature has now been integrated into a Pertamina system called **AIDA** as an automated segmentation module.

#### **Potential Applications**

Segmentation models like this have broad applications across various industries, including:
-   **Infrastructure Inspection**: Automatically detecting corrosion or damage on bridges, pipes, or other metal structures.
-   **Manufacturing**: Identifying product defects on a production line.
-   **Medical Analysis**: Segmenting organs or abnormal tissues from medical images like CT scans or MRIs.
-   **Environmental Monitoring**: Mapping land cover types from satellite imagery.
