package machinelearningcoursework;

import java.util.ArrayList;

/**
 * Entry point for the machine learning pipeline. Orchestrates preprocessing,
 * cross-validation, and evaluation of kNN, SVM, MLP, and a unified classifier.
 */
public class MachineLearningPipeline {
	public static void main(String[] args) {
		// Initialise data and models
		MachineLearningPipeline pipeline = new MachineLearningPipeline();
		System.out.println("Initialising machine learning pipeline...");

		// Load and preprocess data
		System.out.println("Loading datasets...");
		DataPreprocessor dataPreprocessor = new DataPreprocessor();
		ArrayList<int[]> trainingFeaturesDataset1 = new ArrayList<>();
		ArrayList<int[]> trainingFeaturesDataset2 = new ArrayList<>();
		ArrayList<Integer> trainingLabelsDataset1 = new ArrayList<>();
		ArrayList<Integer> trainingLabelsDataset2 = new ArrayList<>();
		pipeline.loadAndPreprocessData(dataPreprocessor, trainingFeaturesDataset1, trainingLabelsDataset1,
				trainingFeaturesDataset2, trainingLabelsDataset2);
		System.out.println("Datasets loaded and preprocessed successfully!");

		// Perform cross-validation and evaluation
		System.out.println("Starting cross-validation and evaluation...");
		pipeline.performCrossValidation(dataPreprocessor, trainingFeaturesDataset1, trainingLabelsDataset1);
		System.out.println("Cross-validation and evaluation completed!");
	}

	/**
	 * Loads and preprocesses datasets by reading from file and normalising
	 * features.
	 *
	 * @param dataPreprocessor The data preprocessor instance to handle
	 *                         preprocessing.
	 * @param featuresDataset1 List to store feature rows of dataset 1.
	 * @param labelsDataset1   List to store labels of dataset 1.
	 * @param featuresDataset2 List to store feature rows of dataset 2.
	 * @param labelsDataset2   List to store labels of dataset 2.
	 */
	private void loadAndPreprocessData(DataPreprocessor dataPreprocessor, ArrayList<int[]> featuresDataset1,
			ArrayList<Integer> labelsDataset1, ArrayList<int[]> featuresDataset2, ArrayList<Integer> labelsDataset2) {
		String dataset1Path = "dataSet1.csv"; // File path for dataset 1
		String dataset2Path = "dataSet2.csv"; // File path for dataset 2

		// Load features and labels from dataset files
		System.out.println("Loading dataset 1 from " + dataset1Path + "...");
		dataPreprocessor.loadFileData(dataset1Path, featuresDataset1, labelsDataset1, true); // Debugging disabled
		System.out.println("Dataset 1 loaded successfully!");

		System.out.println("Loading dataset 2 from " + dataset2Path + "...");
		dataPreprocessor.loadFileData(dataset2Path, featuresDataset2, labelsDataset2, true); // Debugging disabled
		System.out.println("Dataset 2 loaded successfully!");

		// Normalise features using L2 normalisation for both datasets
		System.out.println("Normalising features for both datasets...");
		dataPreprocessor.normaliseFeatures(featuresDataset1);
		dataPreprocessor.normaliseFeatures(featuresDataset2);
		System.out.println("Feature normalisation completed!");
	}

	/**
	 * Performs cross-validation and evaluates classifiers on the dataset.
	 *
	 * @param dataPreprocessor Preprocessor for handling data-related tasks.
	 * @param features         List of feature rows from the dataset.
	 * @param labels           List of labels from the dataset.
	 */
	private void performCrossValidation(DataPreprocessor dataPreprocessor, ArrayList<int[]> features,
			ArrayList<Integer> labels) {
		System.out.println("\n--- Evaluation Results ---"); // Log the start of evaluation phase

		// Prepare two-fold datasets for cross-validation
		ArrayList<int[]> fold1Features = new ArrayList<>();
		ArrayList<int[]> fold2Features = new ArrayList<>();
		ArrayList<Integer> fold1Labels = new ArrayList<>();
		ArrayList<Integer> fold2Labels = new ArrayList<>();
		dataPreprocessor.splitDataset(features, labels, fold1Features, fold1Labels, fold2Features, fold2Labels);

		// Convert labels for SVM binary classification
		dataPreprocessor.convertLabelsForSVM(fold1Labels, 1); // Convert target class labels for Fold 1
		dataPreprocessor.convertLabelsForSVM(fold2Labels, 1); // Convert target class labels for Fold 2

		// Initialise evaluation and validation framework
		ClassifierEvaluator evaluator = new ClassifierEvaluator();
		TwoFoldValidation twoFoldValidation = new TwoFoldValidation(evaluator);

		// Evaluate kNN
		System.out.println("Evaluating kNN classifier... \n ");
		evaluateKNN(twoFoldValidation, fold1Features, fold1Labels, fold2Features, fold2Labels);

		// Evaluate SVM
		System.out.println("Evaluating SVM classifier...  \n");
		SupportVectorMachineClassifier svm = new SupportVectorMachineClassifier(64, 0.005, 1.0, 10000);
		evaluateSVM(twoFoldValidation, svm, fold1Features, fold1Labels, fold2Features, fold2Labels);

		// Evaluate MLP
		System.out.println("Evaluating MLP classifier...  \n");
		MultiLayerPerceptronClassifier mlp = new MultiLayerPerceptronClassifier(64, 128, 1, 0.0001, 2000);
		evaluateMLP(twoFoldValidation, mlp, fold1Features, fold1Labels, fold2Features, fold2Labels);

		// Evaluate Unified Classifier
		System.out.println("Evaluating Unified Classifier...  \n");
		evaluateUnifiedClassifier(evaluator, svm, mlp, fold1Features, fold1Labels, fold2Features, fold2Labels);
	}

	/**
	 * Evaluates the k-Nearest Neighbors (kNN) classifier and prints its accuracy.
	 */
	private void evaluateKNN(TwoFoldValidation twoFoldValidation, ArrayList<int[]> fold1Features,
			ArrayList<Integer> fold1Labels, ArrayList<int[]> fold2Features, ArrayList<Integer> fold2Labels) {
		KNearestNeighbors knn = new KNearestNeighbors(); // Instantiate kNN classifier
		double accuracy = twoFoldValidation.runTwoFoldValidation(knn, fold1Features, fold1Labels, fold2Features,
				fold2Labels);
		System.out.printf("kNN Overall Accuracy (Two-Fold Test): %.2f%%\n ", accuracy);
	}

	/**
	 * Evaluates the Support Vector Machine (SVM) classifier and prints its
	 * accuracy.
	 */
	private void evaluateSVM(TwoFoldValidation twoFoldValidation, SupportVectorMachineClassifier svm,
			ArrayList<int[]> fold1Features, ArrayList<Integer> fold1Labels, ArrayList<int[]> fold2Features,
			ArrayList<Integer> fold2Labels) {
		double accuracy = twoFoldValidation.runTwoFoldValidation(svm, fold1Features, fold1Labels, fold2Features,
				fold2Labels);
		System.out.printf("SVM Overall Accuracy (Two-Fold Test): %.2f%%\n", accuracy);
	}

	/**
	 * Evaluates the Multi-Layer Perceptron (MLP) classifier and prints its
	 * accuracy.
	 */
	private void evaluateMLP(TwoFoldValidation twoFoldValidation, MultiLayerPerceptronClassifier mlp,
			ArrayList<int[]> fold1Features, ArrayList<Integer> fold1Labels, ArrayList<int[]> fold2Features,
			ArrayList<Integer> fold2Labels) {
		double accuracy = twoFoldValidation.runTwoFoldValidation(mlp, fold1Features, fold1Labels, fold2Features,
				fold2Labels);
		System.out.printf("MLP Overall Accuracy (Two-Fold Test): %.2f%%\n", accuracy);
	}

	/**
	 * Evaluates a unified classifier combining kNN, SVM, and MLP predictions and
	 * prints its accuracy.
	 */
	private void evaluateUnifiedClassifier(ClassifierEvaluator evaluator, SupportVectorMachineClassifier svm,
			MultiLayerPerceptronClassifier mlp, ArrayList<int[]> fold1Features, ArrayList<Integer> fold1Labels,
			ArrayList<int[]> fold2Features, ArrayList<Integer> fold2Labels) {
		// Evaluate accuracy for Fold 1
		double accuracyFold1 = evaluator.evaluateUnifiedClassifier(fold1Features, fold1Labels, fold2Features,
				fold2Labels, svm, mlp);

		// Evaluate accuracy for Fold 2
		double accuracyFold2 = evaluator.evaluateUnifiedClassifier(fold2Features, fold2Labels, fold1Features,
				fold1Labels, svm, mlp);

		// Print unified classifier results
		System.out.printf("Unified Classifier Accuracy (Fold 1): %.2f%%\n", accuracyFold1);
		System.out.printf("Unified Classifier Accuracy (Fold 2): %.2f%%\n", accuracyFold2);
		System.out.printf("Unified Classifier Overall Accuracy (Two-Fold Test): %.2f%%\n",
				(accuracyFold1 + accuracyFold2) / 2);
	}
}
