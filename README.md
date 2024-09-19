# Network Intrusion Detection Project

## Introduction

This project implements a Network Intrusion Detection System (NIDS) using various machine learning algorithms to classify network traffic as normal or malicious. The models used include Logistic Regression, Gaussian Naive Bayes, Support Vector Machine, Gradient Boosting, Decision Tree Classifier, Neural Networks, K-Means Clustering, and Random Forest. The goal is to identify the most effective model based on accuracy, precision, recall, F1 score, and execution time.

## Table of Contents

- [Project Overview](#project-overview)
- [Algorithms Used](#algorithms-used)
- [Data Preprocessing](#data-preprocessing)
- [Model Evaluation](#model-evaluation)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

Network Intrusion Detection Systems are critical for identifying and mitigating unauthorized access or attacks on a network. This project explores various machine learning models to enhance the detection rate and minimize false positives. Each algorithm is trained and evaluated on a labeled dataset, comparing performance to identify the optimal model.

## Algorithms Used

1. **Logistic Regression**: A statistical method for binary classification problems.
2. **Gaussian Naive Bayes**: A probabilistic classifier based on Bayes' theorem with the assumption of Gaussian distribution of features.
3. **Support Vector Machine (SVM)**: A supervised learning model that finds the optimal hyperplane to classify data points.
4. **Gradient Boosting**: An ensemble technique that builds models sequentially to minimize error.
5. **Decision Tree Classifier**: A tree-structured classifier that splits data based on feature values.
6. **Neural Network**: A deep learning model composed of multiple layers of neurons that learn patterns from data.
7. **K-Means Clustering**: An unsupervised algorithm that groups data points into clusters.
8. **Random Forest**: An ensemble method using multiple decision trees to improve classification accuracy.

## Data Preprocessing

The dataset used in this project is preprocessed to remove missing values, scale features, and encode categorical variables as needed. Ensure your dataset is prepared similarly to avoid discrepancies in model performance.

## Model Evaluation

Each model is evaluated based on the following metrics:

- **Accuracy**: The percentage of correct predictions.
- **Precision**: The number of true positive results divided by the number of all positive results.
- **Recall**: The number of true positive results divided by the number of positives in actual data.
- **F1 Score**: The harmonic mean of precision and recall, balancing the two metrics.
- **Execution Time**: The time taken to train and test each model.

Cross-validation is used to provide a more reliable estimate of model performance.

## Installation

To run this project, you need to have Python installed along with the necessary libraries. You can install the required packages using:

```bash
pip install -r requirements.txt
