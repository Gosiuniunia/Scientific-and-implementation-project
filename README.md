# Personal Colour Analysis System

This repository contains the implementation of a **Personal Colour Analysis (PCoA)** system.  
The goal of this project is to classify individuals into seasonal color types (Spring, Summer, Autumn, Winter) based on image data.

## 🔍 Project Overview

The focus of this project is on building and comparing different **Machine Learning (ML)** and **Deep Learning (DL)** models for personal colour classification.

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