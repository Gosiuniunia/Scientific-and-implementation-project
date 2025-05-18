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
     - Processing of raw facial images using CNN
     - Integration of pre-implemented image augmentation methods, including:
       - Geometric transformations
       - GANs

- Model evaluation and comparison using accuracy, precision, recall and f1 score with statistical testing