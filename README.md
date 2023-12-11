# Machine Learning Models for Classification
This repository contains a Jupyter Notebook (Learning ML.ipynb) that demonstrates the implementation of several machine learning models for classification tasks. The models are applied to a dataset related to high-energy gamma-ray astronomy. The code is developed using Google Colaboratory.

# Dataset
The dataset contains features such as fLength, fWidth, fSize, fConc, fConcl, fAsym, fM3Long, fM3Trans, fAlpha, fDist, and a target variable 'class'. The target variable is binary, indicating whether an observed event is a gamma-ray ('g') or a hadron ('h').

# Data Exploration
The notebook includes data exploration visualizations, such as histograms for each feature, comparing the distributions of gamma-ray and hadron classes.

# Data Preprocessing
The dataset is split into training, validation, and test sets. The features are scaled using StandardScaler, and an option for oversampling the minority class is provided.

# Classification Models
The following classification models are implemented and evaluated on the test set:

**k-Nearest Neighbors (kNN):**

Utilizes the KNeighborsClassifier from scikit-learn.
Prints the classification report.

**Naive Bayes:**

Implements Gaussian Naive Bayes using GaussianNB from scikit-learn.
Prints the classification report.

**Logistic Regression:**

Applies logistic regression using LogisticRegression from scikit-learn.
Prints the classification report.

**Support Vector Machines (SVM):**

Implements a Support Vector Machine classifier using SVC from scikit-learn.
Prints the classification report.

**Neural Network:**

Implements a simple neural network using TensorFlow/Keras.
The architecture includes multiple layers with configurable parameters.
Training hyperparameters (number of nodes, dropout probability, learning rate, batch size, and epochs) are systematically explored to optimize model performance.
Displays training and validation loss/accuracy plots.

# Usage
Open the Jupyter Notebook Learning ML.ipynb in a compatible environment (e.g., Google Colab).
Execute each cell sequentially to load the dataset, preprocess the data, and train/evaluate various machine learning models.
Feel free to experiment with different hyperparameter values and extend the analysis as needed. If using a different dataset, ensure the data format and preprocessing steps align with the requirements of the models implemented in this notebook.
