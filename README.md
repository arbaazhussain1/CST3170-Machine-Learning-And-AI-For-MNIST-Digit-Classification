#CST3170-Machine-Learning-And-AI-For-MNIST-Digit-Classification

This project implements a machine learning pipeline for digit classification using the MNIST dataset. It includes multiple classification algorithms (kNN, SVM, MLP) and supports evaluation using cross-validation. A unified classifier combining predictions from different models is also implemented for improved accuracy.

---

## Features

### 1. **Classification Algorithms**
- **k-Nearest Neighbors (kNN):**
  - Predicts class labels using the nearest neighbors based on Euclidean distance.
  - Configurable parameter: `numNeighbors`.
- **Support Vector Machine (SVM):**
  - Multi-class classification using a one-vs-all strategy and hinge loss optimization.
  - Configurable hyperparameters: `learningRate`, `regularisationParam`, `maxIterations`.
- **Multi-Layer Perceptron (MLP):**
  - Neural network with a single hidden layer.
  - Supports softmax activation for multi-class classification.
  - Configurable parameters: `hiddenLayerSize`, `learningRate`, `maxTrainingIterations`.

### 2. **Pipeline Workflow**
- **Preprocessing:** Load and preprocess MNIST data from CSV files.
- **Cross-Validation:** Two-fold cross-validation for robust model evaluation.
- **Evaluation Metrics:** Accuracy metrics and confusion matrices for individual and combined classifiers.
- **Unified Classifier:** Combines predictions from kNN, SVM, and MLP using meta-features and trains a meta-classifier.

### 3. **Extensibility**
- Designed for easy integration of additional classifiers or datasets.
- Modular architecture ensures maintainability and scalability.

---

## Code Structure

### 1. **Core Classes**
- **`DataPreprocessor`:**
  - Handles loading datasets, normalizing features, and splitting data.
  - Reads MNIST data from CSV files.
- **`KNearestNeighbors`:**
  - Implements the kNN algorithm with configurable number of neighbors.
- **`SupportVectorMachineClassifier`:**
  - Implements SVM using gradient descent for one-vs-all classification.
- **`MultiLayerPerceptronClassifier`:**
  - Implements a single-hidden-layer MLP with backpropagation and softmax.
- **`ClassifierEvaluator`:**
  - Calculates accuracy and generates confusion matrices.
  - Supports unified evaluation of multiple classifiers.
- **`TwoFoldValidation`:**
  - Manages two-fold cross-validation for any supported classifier.

### 2. **Entry Point**
- **`MachineLearningPipeline`:**
  - Coordinates data preprocessing, cross-validation, and classifier evaluation.
  - Entry point for the application.

---

## Usage

### 1. **Dataset Preparation**
Ensure the MNIST data is available as two CSV files:
- `dataSet1.csv`
- `dataSet2.csv`

Each file should have 64 feature columns followed by a label column.

### 2. **Execution**
Run the `MachineLearningPipeline` class to:
1. Load datasets.
2. Preprocess data.
3. Perform cross-validation.
4. Evaluate classifiers (kNN, SVM, MLP, Unified).

### 3. **Configurations**
Modify constants in the relevant classes to fine-tune the pipeline:
- `KNN_NEIGHBORS` (kNN)
- `SVM_LEARNING_RATE`, `SVM_REGULARISATION_PARAM`, `SVM_MAX_TRAINING_ITERATIONS` (SVM)
- `MLP_HIDDEN_LAYER_SIZE`, `MLP_LEARNING_RATE`, `MLP_MAX_TRAINING_ITERATIONS` (MLP)

---

## Output
The application prints:
- Accuracy for each fold during cross-validation.
- Overall accuracy for each classifier.
- Confusion matrices for evaluation.

---

## Dependencies
- Java JDK 8 or later.
- No external libraries are required.

---

## Authors
- [Your Name or Team Name]

## License
This project is licensed under the MIT License. See the LICENSE file for details.

