package machinelearningcoursework;

import java.util.ArrayList;
import java.util.List;

public class MachineLearningPipeline {

	public static void main(String[] args) {
		// Paths to dataset files
		String dataset1Path = "dataSet1.csv"; // Path to the first dataset
		String dataset2Path = "dataSet2.csv"; // Path to the second dataset

		// Data structures to hold features and labels for both datasets
		List<int[]> trainingFeaturesDataset1 = new ArrayList<>();
		List<int[]> trainingFeaturesDataset2 = new ArrayList<>();
		List<Integer> trainingLabelsDataset1 = new ArrayList<>();
		List<Integer> trainingLabelsDataset2 = new ArrayList<>();

		// Instantiate the DataPreprocessor for handling preprocessing tasks
		DataPreprocessor dataPreprocessor = new DataPreprocessor();

		// Load the datasets into feature and label lists
		dataPreprocessor.loadFileData(dataset1Path, trainingFeaturesDataset1, trainingLabelsDataset1);
		dataPreprocessor.loadFileData(dataset2Path, trainingFeaturesDataset2, trainingLabelsDataset2);

		// Normalise the features to ensure consistent scaling
		dataPreprocessor.normaliseFeatures(trainingFeaturesDataset1);
		dataPreprocessor.normaliseFeatures(trainingFeaturesDataset2);

		// Prepare data structures for cross-validation folds
		List<int[]> fold1Features = new ArrayList<>();
		List<int[]> fold2Features = new ArrayList<>();
		List<Integer> fold1Labels = new ArrayList<>();
		List<Integer> fold2Labels = new ArrayList<>();

		// Split dataset1 into two folds for cross-validation
		dataPreprocessor.splitDataset(trainingFeaturesDataset1, trainingLabelsDataset1, fold1Features, fold1Labels,
				fold2Features, fold2Labels);

		// Convert labels for SVM (binary classification: target class = 1, others = -1)
		dataPreprocessor.convertLabelsForSVM(fold1Labels, 1);
		dataPreprocessor.convertLabelsForSVM(fold2Labels, 1);

		// Instantiate the ClassifierEvaluator to handle evaluation tasks
		ClassifierEvaluator classifierEvaluator = new ClassifierEvaluator();
		System.out.println("\n--- Evaluation Results ---");

		// k-Nearest Neighbors (kNN) evaluation
		KNearestNeighbors knn = new KNearestNeighbors();
		double accuracyOfKNN = classifierEvaluator.evaluateKNN(knn, fold1Features, fold1Labels, fold2Features,
				fold2Labels);
		System.out.printf("kNN Accuracy: %.2f%%\n", accuracyOfKNN);

		// Support Vector Machine (SVM) evaluation with grid search with sigmoid kernel
		// for hyper-parameter tuning
		SupportVectorMachineClassifier svm = new SupportVectorMachineClassifier(64, 0.005, 1.0, 10000);
		svm.gridSearchWithSigmoidKernel(fold1Features, fold1Labels, fold2Features, fold2Labels);
		double accuracyOfSVM = svm.evaluate(fold2Features, fold2Labels);
		System.out.printf("SVM Accuracy: %.2f%%\n", accuracyOfSVM);

		// Multi-Layer Perceptron (MLP) evaluation
		MultiLayerPerceptronClassifier mlp = new MultiLayerPerceptronClassifier(64, 128, 1, 0.0001, 2000);
		mlp.train(fold1Features, fold1Labels); // Train the MLP on fold1
		double accuracyOfMLP = mlp.evaluate(fold2Features, fold2Labels); // Evaluate the MLP on fold2
		System.out.printf("MLP Accuracy: %.2f%%\n", accuracyOfMLP);

		// Unified Classifier evaluation (stacking ensemble method)
		double accuracyOfUnifiedClassifier = classifierEvaluator.evaluateUnifiedClassifier(fold1Features, fold1Labels,
				fold2Features, fold2Labels, svm, mlp);
		System.out.printf("Unified Classifier Accuracy: %.2f%%\n", accuracyOfUnifiedClassifier);
	}
}
