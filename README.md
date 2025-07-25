# Fashion-MNIST Classification with PCA and Ensemble Learning

This project implements classification of the [Fashion-MNIST](https://www.kaggle.com/datasets/zalando-research/fashionmnist) dataset using dimensionality reduction (PCA), hyperparameter tuning (GridSearchCV), and ensemble learning (Voting Classifier) in Python with scikit-learn.

## 📂 Dataset

- `fashion-mnist_train.csv`: Training data (images + labels)
- `fashion-mnist_test.csv`: Test data (images + labels)

Each row represents a 28×28 grayscale image (784 pixel values) and a corresponding label (0–9).

## 🛠️ Libraries Used

- `pandas`, `numpy`: Data manipulation
- `scikit-learn`: Machine learning pipeline, PCA, classification models, and evaluation

## 📈 Workflow

### 1. Data Preparation
- Load CSV files into pandas DataFrames
- Separate features and labels for train and test sets

### 2. Dimensionality Reduction
- Apply **PCA** to reduce dimensions while retaining **90% variance**

### 3. Model Selection & Tuning
- Define 3 classifiers with hyperparameter grids:
  - **Logistic Regression**
  - **Random Forest**
  - **K-Nearest Neighbors**
- Use `Pipeline` and `GridSearchCV` for each model with 3-fold cross-validation
- Record best scores and parameters

### 4. Ensemble Learning
- Combine the top-performing models using `VotingClassifier` (hard voting)
- Train the ensemble on full training data
- Evaluate on test set

## 🧪 Results

| Model                | CV Accuracy | Test Accuracy |
|---------------------|-------------|---------------|
| Logistic Regression | ~0.87       | ~0.87         |
| Random Forest       | ~0.88       | ~0.88         |
| KNN                 | ~0.87       | ~0.87         |
| **Voting Classifier** | -           | **~0.89**     |

> Accuracy improved by combining models using a Voting Classifier.

## 📊 Evaluation Metrics

- Accuracy Score
- Confusion Matrix
- Classification Report (Precision, Recall, F1-Score)
