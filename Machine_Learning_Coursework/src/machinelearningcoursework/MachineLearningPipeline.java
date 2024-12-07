package machinelearningcoursework;

import java.util.ArrayList;
import java.util.List;

public class MachineLearningPipeline {

	public static void main(String[] args) {
		// Paths to dataset files
		String dataset1Path = "dataSet1.csv"; // Path to the first dataset
		String dataset2Path = "dataSet2.csv"; // Path to the second dataset

		// Data structures to hold features and labels for both datasets
		List<int[]> trainingFeaturesDataset1 = new ArrayList<>(); // Features for dataset 1
		List<int[]> trainingFeaturesDataset2 = new ArrayList<>(); // Features for dataset 2
		List<Integer> trainingLabelsDataset1 = new ArrayList<>(); // Labels for dataset 1
		List<Integer> trainingLabelsDataset2 = new ArrayList<>(); // Labels for dataset 2

		// Instantiate the DataPreprocessor for handling preprocessing tasks
		DataPreprocessor dataPreprocessor = new DataPreprocessor();

		// Load the datasets into feature and label lists
		dataPreprocessor.loadFileData(dataset1Path, trainingFeaturesDataset1, trainingLabelsDataset1);
		dataPreprocessor.loadFileData(dataset2Path, trainingFeaturesDataset2, trainingLabelsDataset2);

		// Normalise the features to ensure consistent scaling
		dataPreprocessor.normaliseFeatures(trainingFeaturesDataset1); // Normalise dataset 1
		dataPreprocessor.normaliseFeatures(trainingFeaturesDataset2); // Normalise dataset 2

		// Prepare data structures for cross-validation folds
		List<int[]> fold1Features = new ArrayList<>(); // Fold 1 features
		List<int[]> fold2Features = new ArrayList<>(); // Fold 2 features
		List<Integer> fold1Labels = new ArrayList<>(); // Fold 1 labels
		List<Integer> fold2Labels = new ArrayList<>(); // Fold 2 labels

		// Split dataset1 into two folds for cross-validation
		dataPreprocessor.splitDataset(trainingFeaturesDataset1, trainingLabelsDataset1, fold1Features, fold1Labels,
				fold2Features, fold2Labels);

		// Convert labels for SVM (binary classification: target class = 1, others = -1)
		dataPreprocessor.convertLabelsForSVM(fold1Labels, 1); // Convert fold 1 labels
		dataPreprocessor.convertLabelsForSVM(fold2Labels, 1); // Convert fold 2 labels

		// Instantiate the ClassifierEvaluator to handle evaluation tasks
		ClassifierEvaluator classifierEvaluator = new ClassifierEvaluator();
		System.out.println("\n--- Evaluation Results ---");

		// k-Nearest Neighbors (kNN) evaluation
		KNearestNeighbors knn = new KNearestNeighbors();

		// Fold 1: Train on Fold 1, Test on Fold 2
		double accuracyKNN_Fold1 = classifierEvaluator.evaluateKNN(knn, fold1Features, fold1Labels, fold2Features,
				fold2Labels);

		// Fold 2: Train on Fold 2, Test on Fold 1
		double accuracyKNN_Fold2 = classifierEvaluator.evaluateKNN(knn, fold2Features, fold2Labels, fold1Features,
				fold1Labels);

		// Print kNN Results
		System.out.printf("kNN Accuracy (Fold 1): %.2f%%\n", accuracyKNN_Fold1); // Accuracy for fold 1
		System.out.printf("kNN Accuracy (Fold 2): %.2f%%\n", accuracyKNN_Fold2); // Accuracy for fold 2
		System.out.printf("kNN Overall Accuracy (Two-Fold Test): %.2f%%\n",
				(accuracyKNN_Fold1 + accuracyKNN_Fold2) / 2);

		// Support Vector Machine (SVM) evaluation
		SupportVectorMachineClassifier svm = new SupportVectorMachineClassifier(64, 0.005, 1.0, 10000);

		// Fold 1: Train on Fold 1, Test on Fold 2
		svm.train(fold1Features, fold1Labels); // Train SVM on fold 1
		double accuracySVM_Fold1 = svm.evaluate(fold2Features, fold2Labels); // Evaluate SVM on fold 2

		// Fold 2: Train on Fold 2, Test on Fold 1
		svm.train(fold2Features, fold2Labels); // Train SVM on fold 2
		double accuracySVM_Fold2 = svm.evaluate(fold1Features, fold1Labels); // Evaluate SVM on fold 1

		// Print SVM Results
		System.out.printf("SVM Accuracy (Fold 1): %.2f%%\n", accuracySVM_Fold1); // Accuracy for fold 1
		System.out.printf("SVM Accuracy (Fold 2): %.2f%%\n", accuracySVM_Fold2); // Accuracy for fold 2
		System.out.printf("SVM Overall Accuracy (Two-Fold Test): %.2f%%\n",
				(accuracySVM_Fold1 + accuracySVM_Fold2) / 2);

		// Multi-Layer Perceptron (MLP) evaluation
		MultiLayerPerceptronClassifier mlp = new MultiLayerPerceptronClassifier(64, 128, 1, 0.0001, 2000);

		// Fold 1: Train on Fold 1, Test on Fold 2
		mlp.train(fold1Features, fold1Labels); // Train MLP on fold 1
		double accuracyMLP_Fold1 = mlp.evaluate(fold2Features, fold2Labels); // Evaluate MLP on fold 2

		// Fold 2: Train on Fold 2, Test on Fold 1
		mlp.train(fold2Features, fold2Labels); // Train MLP on fold 2
		double accuracyMLP_Fold2 = mlp.evaluate(fold1Features, fold1Labels); // Evaluate MLP on fold 1

		// Print MLP Results
		System.out.printf("MLP Accuracy (Fold 1): %.2f%%\n", accuracyMLP_Fold1); // Accuracy for fold 1
		System.out.printf("MLP Accuracy (Fold 2): %.2f%%\n", accuracyMLP_Fold2); // Accuracy for fold 2
		System.out.printf("MLP Overall Accuracy (Two-Fold Test): %.2f%%\n",
				(accuracyMLP_Fold1 + accuracyMLP_Fold2) / 2);

		// Unified Classifier evaluation (stacking ensemble method)
		double accuracyUnified_Fold1 = classifierEvaluator.evaluateUnifiedClassifier(fold1Features, fold1Labels,
				fold2Features, fold2Labels, svm, mlp); // Unified classifier accuracy for fold 1
		double accuracyUnified_Fold2 = classifierEvaluator.evaluateUnifiedClassifier(fold2Features, fold2Labels,
				fold1Features, fold1Labels, svm, mlp); // Unified classifier accuracy for fold 2

		// Print Unified Classifier Results
		System.out.printf("Unified Classifier Accuracy (Fold 1): %.2f%%\n", accuracyUnified_Fold1);
		System.out.printf("Unified Classifier Accuracy (Fold 2): %.2f%%\n", accuracyUnified_Fold2);
		System.out.printf("Unified Classifier Overall Accuracy (Two-Fold Test): %.2f%%\n",
				(accuracyUnified_Fold1 + accuracyUnified_Fold2) / 2); // Overall accuracy
	}

}
