# Personal Colour Analysis System

ins implementations of machine learning (ML) and deep learning (DL) methods for **Personal Colour Analysis (PCoA)** system.  
The goal of this project is to classify individuals into seasonal color types (Spring, Summer, Autumn, Winter) based on image data.

## 🔍 Project Overview

The environment incorporates various components including color feature extraction (based on facial images and landmark detection), hyperparameter tuning of ML classifiers (KNN, SVM, Decision Trees), and preparation and training of DL models (e.g., VGG16), with support for data augmentation.

Additionally, the environment provides tools for testing and comparing classifier performance using statistical tests.

## Key Features
- Image preprocessing including **White balancing**
- Two distinct modelling approaches:

  1. **Feature-based Machine Learning**:
     - Extraction of dominant colours from key facial regions (eyes, skin, eyebrows) using MediaPipe facial landmarks
     - Application of classical ML algorithms:
       - Support Vector Machine (SVM)
       - Decision Tree (DT)
       - k-Nearest Neighbors (KNN)
     - Hyperparameter tuning for optimal performance of ML models

  2. **End-to-End Deep Learning**:
     - Processing of raw facial images using CNN VGG16 model
     - Integration of pre-implemented model free image augmentation methods

- Model evaluation and comparison using accuracy, precision, recall and f1 score with statistical testing

## Usage
### 1. Clone the repository

```bash
git clone 
```
### 2. Install dependencies

```bash
pip install -r requirements.txt
```

###  3. Configure paths in config.ini

Open `config.ini` and update the paths to match your environment:

- `mediapipe_model_path` - download the `.task` model file from the official MediaPipe website: https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker?hl=pl

- `dataset_path ` - path to the dataset folder containing subfolders, each subfolder represents a different class label (personal season)

- `extracted_features_csv_file_path  ` path and filename for the output CSV file where the extracted features will be saved

- `model_free_augmented_images_path` - path to the dataset folder which will contain model-free augmented images. Folder should contain subfolders, each subfolder represents a different class label (personal season)

- `folds_assignment_path` - path to the csv file with images assigned to training folds - used when model training without data augmentation is performed

- `model_free_folds_assignment_path` - path to the csv file with images assigned to training folds - used when model training with data augmentation is performed

- `target_size` - size of the image for training. For VGG16 model the value 224 is required

- `offset` - parameter used to perform multiple experiment runs. Should be set to 0 in the first trial and to 5 if the second run is done

- `current_approach` - parameter used to specify current approach to train the model

- `summary_scores_path` - path to the txt file, where the summary of all scores will be saved

- `testing_path` - path to the txt file, where result of the statistical testing will be saved

- `plot_path` - path to the png file, where bar plot of scores will be saved

### 4. Run main.py file


```bash
python main.py
```
Results of the experiment will be saved in `scores` folder.