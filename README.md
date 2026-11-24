# Facial Emotion Recognition Using Hybrid CNN and Facial Muscle Features

## Project Overview

This project implements a **Facial Emotion Recognition (FER)** system using a hybrid approach that combines deep learning-based image features and facial muscle activation features.

The system uses:
- **MobileNetV2** for extracting deep facial image features.
- **MediaPipe Face Mesh** for detecting facial landmarks.
- **Facial Muscle Feature Extraction** for identifying geometric changes in facial expressions.
- **XGBoost** for emotion classification.

The hybrid model combines CNN features with facial muscle features to recognize seven different human emotions.

## Emotions Classified

The model classifies facial expressions into seven categories:

1. Anger
2. Contempt
3. Disgust
4. Fear
5. Happy
6. Sadness
7. Surprise

## Technologies Used

- Python
- TensorFlow / Keras
- MobileNetV2
- MediaPipe
- OpenCV
- XGBoost
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Google Colab

## Model Architecture

The project follows a hybrid feature extraction and classification approach.

### 1. CNN Feature Extraction

- Uses pretrained MobileNetV2 with ImageNet weights.
- Removes the final classification layer.
- Uses Global Average Pooling to generate a 1280-dimensional feature vector.
- Input image size: 160 × 160 pixels.

### 2. Facial Muscle Feature Extraction

MediaPipe Face Mesh detects facial landmarks, which are used to calculate eight facial muscle-related features:

- Inner Brow Raise
- Outer Brow Raise
- Brow Lower
- Upper Lip Raise
- Jaw Drop
- Lip Stretch
- Smile
- Eye Close

These features capture geometric variations in facial expressions.

### 3. Hybrid Feature Fusion

The extracted features are combined:

- CNN features: 1280 dimensions
- Facial muscle features: 8 dimensions
- Total hybrid feature vector: **1288 dimensions**

### 4. Emotion Classification

XGBoost classifiers are trained using two approaches:

- CNN-only classification.
- Hybrid CNN + facial muscle feature classification.

## Project Workflow

1. Load and preprocess the facial expression dataset.
2. Detect facial landmarks using MediaPipe Face Mesh.
3. Extract eight facial muscle-related features.
4. Resize images to 160 × 160 pixels.
5. Apply LAB luminance histogram equalization.
6. Extract deep features using MobileNetV2.
7. Standardize facial muscle features.
8. Combine CNN and facial muscle features.
9. Train XGBoost classifiers.
10. Evaluate model performance using accuracy, classification reports and confusion matrices.
11. Predict emotions from new facial images.

## Dataset Structure

The dataset contains seven emotion categories.

```text
dataset/
│
├── anger/
├── contempt/
├── disgust/
├── fear/
├── happy/
├── sadness/
└── surprise/
```

Supported image formats:
- JPG
- JPEG
- PNG

The notebook uses a ZIP archive containing these emotion folders.

## Model Configuration

| Parameter | Value |
|---|---|
| CNN Architecture | MobileNetV2 |
| Pretrained Weights | ImageNet |
| Input Image Size | 160 × 160 |
| CNN Feature Dimensions | 1280 |
| Facial Muscle Features | 8 |
| Hybrid Feature Dimensions | 1288 |
| Classification Algorithm | XGBoost |
| Train-Test Split | 80:20 |
| Random State | 42 |

### XGBoost Parameters

**CNN-only Model**
- n_estimators = 200
- max_depth = 5
- learning_rate = 0.05
- subsample = 0.9
- colsample_bytree = 0.9

**Hybrid Model**
- n_estimators = 300
- max_depth = 6
- learning_rate = 0.05
- subsample = 0.9
- colsample_bytree = 0.9

## Results and Performance

The following results were obtained from the notebook's test dataset:

| Model | Accuracy |
|---|---:|
| CNN-only Model | 96.45% |
| Hybrid CNN + Facial Muscle Model | 95.43% |

The test dataset contained 197 images.

The notebook also generates:
- Confusion Matrix
- Classification Reports
- Feature Importance Graphs
- Facial Muscle Feature Correlation Heatmap
- PCA Visualizations
- Per-Class Accuracy Graph
- Class Distribution Plot

## Prediction

The notebook provides two prediction functions:

**CNN-only prediction**
```python
predict_cnn_only(image_path)
```

**Hybrid prediction**
```python
predict_hybrid(image_path)
```

The hybrid prediction function uses facial landmarks when available and falls back to the CNN-only classifier if facial landmarks cannot be detected.

The prediction output includes:
- Predicted emotion
- Prediction confidence
- Prediction mode (Hybrid or CNN-only fallback)

## Installation and Execution

### Google Colab (Recommended)

1. Upload `FER_Hybrid_Final.ipynb` to Google Colab.
2. Run the notebook cells sequentially.
3. Upload the dataset ZIP file when prompted.
4. Allow the notebook to perform preprocessing and feature extraction.
5. Train the XGBoost classifiers.
6. Review the evaluation results.
7. Upload a test facial image when prompted to predict its emotion.

### Dependencies

The notebook uses the following libraries:

```bash
pip install mediapipe==0.10.14
pip install xgboost==2.1.0
pip install opencv-python
pip install tensorflow
pip install numpy pandas scikit-learn
pip install matplotlib seaborn
```

## Future Improvements

- Improve hybrid model accuracy.
- Experiment with additional facial features.
- Test the model on larger and more diverse datasets.
- Develop a real-time facial emotion recognition application.
- Build a web-based interface for emotion prediction.
- Save trained models for independent inference.

## Limitations

- Performance depends on image quality and facial landmark detection.
- Facial muscle-related features are geometric approximations rather than direct biological muscle measurements.
- The reported accuracy is based on one train-test split.
- The notebook currently requires model training before prediction.

## Conclusion

This project explores a hybrid approach to facial emotion recognition by combining deep learning-based visual representations with facial landmark-derived geometric features.

It compares CNN-only and hybrid classification approaches using XGBoost and evaluates their performance across seven emotion categories.

---

**Project:** Facial Emotion Recognition (FER)  
**Notebook:** `FER_Hybrid_Final.ipynb`  
**Classification:** Multi-class Emotion Recognition
