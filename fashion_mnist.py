import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

# Load Fashion-MNIST dataset
train = pd.read_csv('fashion-mnist_train.csv')
test = pd.read_csv('fashion-mnist_test.csv')

# Explore the data structure
train.head()

# Split features and target variables
x_train = train.drop(columns='label', axis=1)
y_train = train['label']
x_test = test.drop(columns='label', axis=1)
y_test = test['label']

# Initialize PCA to retain 90% of variance
pca = PCA(n_components=0.9)

# Define models and their hyperparameter grids
models = {
   'LogisticRegression': { 
       'model': LogisticRegression(),
       'param': {
           'clf__C': [0.1],
           'clf__penalty': ['l2'],
           'clf__solver': ['saga'],
           'clf__max_iter': [100000]
       }
   },
   'RandomForestClassifier': {
       'model': RandomForestClassifier(),
       'param': {
           'clf__max_depth': [19, 20, 21],
           'clf__criterion': ['gini', 'entropy']
       }
   },
   'KNeighborsClassifier': {
       'model': KNeighborsClassifier(),
       'param': {
           'clf__n_neighbors': [6, 7, 8],
           'clf__weights': ['distance', 'uniform']
       }
   }
}

# Store grid search results
grid_scores = {}

# Perform grid search for each model
for name, model in models.items():
   # Create pipeline with PCA and classifier
   pipe = Pipeline(steps=[
       ('pca', pca),
       ('clf', model['model'])
   ])
   
   # Setup grid search with cross-validation
   grid = GridSearchCV(
       pipe,
       param_grid=model['param'],
       cv=3,
       scoring='accuracy',
       n_jobs=-1,
       verbose=1
   )
   
   # Train the model
   grid.fit(x_train, y_train)
   
   # Store results
   grid_scores[name] = {
       'best_score': grid.best_score_,
       'best_params': grid.best_params_,
       'grid_object': grid
   }
   
   print(f"{name} - Best CV Score: {grid.best_score_}")

# Display best parameters for each model
for name, result in grid_scores.items():
   print(f"\n{name}:")
   print(f"Score: {result['best_score']:.4f}")
   print("Parameters:")
   for param, value in result['best_params'].items():
       print(f"  {param}: {value}")

# Evaluate models on test set
for name, result in grid_scores.items():
   # Get the best trained model
   best_model = result['grid_object'].best_estimator_
   
   # Make predictions on test set
   y_pred = best_model.predict(x_test)
   
   # Display results
   print(f"\n{'='*20} {name} {'='*20}")
   print(f"Test Accuracy: {best_model.score(x_test, y_test):.4f}")
   print("\nClassification Report:")
   print(classification_report(y_test, y_pred))
