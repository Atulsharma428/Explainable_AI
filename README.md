# Explainable_AI 

# Fish Disease Detection Using Image Processing and Machine Learning

This project aims to detect fish diseases by applying image processing techniques and machine learning classifiers. It uses a dataset of fish images, extracts various features, and applies different classifiers to detect and classify fish diseases. The project also explores the impact of various color spaces on the performance of classifiers.

## Table of Contents
1. [Project Structure](#project-structure)
2. [Project Overview](#project-overview)
3. [Features Extracted](#features-extracted)
4. [Classifiers Used](#classifiers-used)
5. [Project Workflow](#project-workflow)
6. [Results and Discussion](#results-and-discussion)
    - [Dataset 1 Results](#dataset-1-results)
    - [Dataset 2 Results](#dataset-2-results)
    - [Dataset 1 Histogram](#dataset-1-histogram)
    - [Dataset 2 Histogram](#dataset-2-histogram)
    - [Feature importance plot of Dataset 1](#feature_imp)
    - [Feature importance plot of Dataset 1](#feature_imp2)

---

## Project Structure

```bash
Fish Disease Detection/

├── assets/       # Folder containing results and sample images of the project
├── Major.ipynb   # Python script with the classification of first dataset.
├── Major2.ipynb   # (ignore) Python script with the classification of Second dataset.
├── Major3.ipynb   # Python script with the classification of Second dataset.
├── different_bins.ipynb     # Datasets accuracies on different bins size using AdaBoostClassifier
├── feature_importance.ipyn  # Python script with feature importance plots
├── histogram_different_bin_sizes.ipynb   # Python script with histogram plots
├── README.md                   # Project documentation (this file)
```

## Project Overview
This project aims to detect fish diseases by applying image processing techniques and machine learning classifiers. It uses a dataset of fish images, extracts various features, and applies different classifiers to detect and classify fish diseases. The project also explores the impact of various color spaces on the performance of classifiers.

## Features Extracted

The project uses a combination of various image features and transforms to enhance the classification process:

- **Local Binary Pattern (LBP)**: A texture descriptor used to extract patterns from grayscale images.
- **Local Directional Pattern (LDP)**: Extracts edge and texture information.
- **Block-Wise Truncation (BWT)**: Captures block-level intensity information in an image.
- **Hough Transform**: Detects lines and shapes within images.
- **Color Features**: Extracts LBP and LDP features from RGB channels and different color spaces (HSV, LAB, YCbCr).
- ![Sample image in grayscale and feature extraction](https://github.com/Atulsharma428/Explainable_AI/blob/main/assets/grayscale.png)


---

## Classifiers Used

Various machine learning classifiers are trained and evaluated using the extracted features:

- **Support Vector Classifier (SVC)**
- **Random Forest Classifier (RF)**
- **Gradient Boosting Classifier (GB)**
- **AdaBoost Classifier (AB)**
- **Decision Tree Classifier (DT)**
- **Logistic Regression (LR)**

---

## Project Workflow

1. **Loading Dataset**:  
   The dataset is organized into folders based on the disease label. Each folder contains images of fish with the corresponding disease.

2. **Data Augmentation**:  
   Image augmentation is applied to increase the diversity of the dataset. Techniques include rotation, zoom, shear, and flips.

3. **Feature Extraction**:
   - Grayscale images are converted, and feature descriptors like LBP, LDP, BWT, and Hough Transform are applied.
   - Features are also extracted from RGB channels and different color spaces like HSV, LAB, and YCbCr.

4. **Training and Evaluation**:
   - The dataset is split into training and testing sets.
   - Classifiers are trained and evaluated on various feature sets, and performance metrics such as accuracy, precision, recall, and F1 score are computed.

5. **Final Evaluation**:  
   Combined features (LBP, LDP, BWT, Hough) are used for classification, and different classifiers are evaluated.
   
## Results and Discussion
The results illustrate how various color spaces affect the performance of classifiers. The project evaluated the classifiers on two separate datasets across four color spaces: RGB, HSV, LAB, and YCbCr. For each color space, the classifiers' performance was measured using Accuracy (Acc), Precision (Prec), Recall (Rec), and F1 Score (F1).

### Dataset 1 Results 
![Result of dataset-1](https://github.com/Atulsharma428/Explainable_AI/blob/main/assets/Result_table_1.png)
### Dataset 2 Results 
![Result of dataset-2](https://github.com/Atulsharma428/Explainable_AI/blob/main/assets/Result_table_2.png)
### Dataset 1 Histogram plot 
![Histogram plot of dataset-1](https://github.com/Atulsharma428/Explainable_AI/blob/main/assets/Result_dataset_1.png)
### Dataset 2 Histogram plot
![Histogram plot of dataset-2](https://github.com/Atulsharma428/Explainable_AI/blob/main/assets/Result_dataset_2.png)
### Feature importance plot of Dataset 1
![Histogram plot of dataset-1](https://github.com/Atulsharma428/Explainable_AI/blob/main/assets/Feature_importance_1.png)
### Feature importance plot of Dataset 2
![Histogram plot of dataset-2](https://github.com/Atulsharma428/Explainable_AI/blob/main/assets/Feature_importance_2.png)

