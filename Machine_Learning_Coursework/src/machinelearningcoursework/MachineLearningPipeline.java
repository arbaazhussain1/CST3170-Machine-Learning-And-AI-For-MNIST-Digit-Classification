package machinelearningcoursework;

import java.io.File;

/**
 * The entry point of the machine learning pipeline manages data preparation, cross-validation,
 * and evaluation of classifiers such as kNN, SVM, MLP, and a unified model, 
 * ensuring streamlined and efficient execution of the entire workflow.
 */
public class MachineLearningPipeline {

	// Constants for dataset's dimensions
	private static final int DATASET_SIZE = 2810; // Number of rows in each dataset's
	private static final int FEATURE_DIMENSIONS = 64; // Number of features per data point

	// kNN configuration
	private static final int KNN_NEIGHBORS = 3; // Number of nearest neighbors

	// SVM configuration
	private static final int SVM_INPUT_FEATURE_SIZE = 64; // Number of input features
	private static final int SVM_NUM_CLASSES = 10; // Number of classes (0-9)
	private static final double SVM_LEARNING_RATE = 0.005; // Learning rate for SVM
	private static final double SVM_REGULARISATION_PARAM = 1.0; // Regularisation parameter
	private static final int SVM_MAX_TRAINING_ITERATIONS = 10000; // Maximum number of training iterations

	// MLP configuration
	private static final int MLP_INPUT_FEATURE_SIZE = 64; // Number of input features
	private static final int MLP_HIDDEN_LAYER_SIZE = 50; // Number of neurons in the hidden layer
	private static final int MLP_OUTPUT_SIZE = 10; // Number of output neurons (number of classes)
	private static final double MLP_LEARNING_RATE = 0.001; // Learning rate for training
	private static final int MLP_MAX_TRAINING_ITERATIONS = 10000; // Maximum number of training iterations

	public static void main(String[] args) {
		// Create the pipeline and initialise components
		MachineLearningPipeline pipeline = new MachineLearningPipeline();
		System.out.println("Initialising machine learning pipeline...");

		// Load and preprocessing data
		System.out.println("Loading datasets...");
		DataPreprocessor dataPreprocessor = new DataPreprocessor();
		int[][] trainingFeaturesDataset1 = new int[DATASET_SIZE][FEATURE_DIMENSIONS];
		int[][] trainingFeaturesDataset2 = new int[DATASET_SIZE][FEATURE_DIMENSIONS];
		int[] trainingLabelsDataset1 = new int[DATASET_SIZE];
		int[] trainingLabelsDataset2 = new int[DATASET_SIZE];
		pipeline.loadAndPreprocessData(dataPreprocessor, trainingFeaturesDataset1, trainingLabelsDataset1,
				trainingFeaturesDataset2, trainingLabelsDataset2);
		System.out.println("Datasets loaded and preprocessed successfully!");

		// Perform cross-validation and evaluation
		System.out.println("Starting cross-validation and evaluation...");
		pipeline.performCrossValidation(dataPreprocessor, trainingFeaturesDataset1, trainingLabelsDataset1,
				trainingFeaturesDataset2, trainingLabelsDataset2);
		System.out.println("Cross-validation and evaluation completed!");
	}

	// Loads dataset's from CSV files, normalises features, and stores them in arrays
	private void loadAndPreprocessData(DataPreprocessor dataPreprocessor, int[][] featuresDataset1,
			int[] labelsDataset1, int[][] featuresDataset2, int[] labelsDataset2) {
		String fileLocation = System.getProperty("user.dir");
		String dataset1Path = fileLocation + File.separator + "dataSet1.csv"; // File path for dataset's 1
		String dataset2Path = fileLocation + File.separator + "dataSet2.csv"; // File path for dataset's 2

		// Load features and labels from dataset's files
		System.out.println("Loading dataset 1 from " + dataset1Path + "...");
		dataPreprocessor.loadFileData(dataset1Path, featuresDataset1, labelsDataset1, true); // Debugging enabled
//         dataPreprocessor.getLabels(dataset1Path);
		System.out.println("Dataset 1 loaded successfully!");

		System.out.println("Loading dataset 2 from " + dataset2Path + "...");
		dataPreprocessor.loadFileData(dataset2Path, featuresDataset2, labelsDataset2, true); // Debugging enabled
//         dataPreprocessor.getLabels(dataset2Path);
		System.out.println("Dataset 2 loaded successfully!");

	}

	// Trains and evaluates all classifiers using two-fold validation
	private void performCrossValidation(DataPreprocessor dataPreprocessor, int[][] featuresDataset1,
			int[] labelsDataset1, int[][] featuresDataset2, int[] labelsDataset2) {
		System.out.println("\n--- Evaluation Results ---"); // Log the start of evaluation phase

		// Initialise evaluation and validation framework
		ClassifierEvaluator evaluator = new ClassifierEvaluator();
		TwoFoldValidation twoFoldValidation = new TwoFoldValidation(evaluator);

		// Evaluate kNN
		System.out.println("Evaluating kNN classifier... \n");
		evaluateKNN(twoFoldValidation, featuresDataset1, labelsDataset1, featuresDataset2, labelsDataset2);

		// Evaluate SVM
		System.out.println("Evaluating SVM classifier...  \n");
		SupportVectorMachineClassifier svm = new SupportVectorMachineClassifier(SVM_INPUT_FEATURE_SIZE, SVM_NUM_CLASSES,
				SVM_LEARNING_RATE, SVM_REGULARISATION_PARAM, SVM_MAX_TRAINING_ITERATIONS);
		evaluateSVM(twoFoldValidation, svm, featuresDataset1, labelsDataset1, featuresDataset2, labelsDataset2);

		// Evaluate MLP
		System.out.println("Evaluating MLP classifier...  \n");
		MultiLayerPerceptronClassifier mlp = new MultiLayerPerceptronClassifier(MLP_INPUT_FEATURE_SIZE,
				MLP_HIDDEN_LAYER_SIZE, MLP_OUTPUT_SIZE, MLP_LEARNING_RATE, MLP_MAX_TRAINING_ITERATIONS);
		evaluateMLP(twoFoldValidation, mlp, featuresDataset1, labelsDataset1, featuresDataset2, labelsDataset2);

		// Evaluate Unified Classifier
		System.out.println("Evaluating Unified Classifier...  \n");
		evaluateUnifiedClassifier(evaluator, svm, mlp, featuresDataset1, labelsDataset1, featuresDataset2,
				labelsDataset2);
	}

	// Runs two-fold validation for k-Nearest Neighbors (kNN) classifier and prints
	// its accuracy.
	private void evaluateKNN(TwoFoldValidation twoFoldValidation, int[][] featuresDataset1, int[] labelsDataset1,
			int[][] featuresDataset2, int[] labelsDataset2) {
		KNearestNeighbors knn = new KNearestNeighbors(); // Instantiate kNN classifier
		double accuracy = twoFoldValidation.runTwoFoldValidation(knn, featuresDataset1, labelsDataset1,
				featuresDataset2, labelsDataset2, KNN_NEIGHBORS); // Use the constant for k
		System.out.printf("kNN Overall Accuracy (Two-Fold Test): %.2f%%\n", accuracy);
	}

	// Runs two-fold validation for Support Vector Machine (SVM) and prints its
	// accuracy.
	private void evaluateSVM(TwoFoldValidation twoFoldValidation, SupportVectorMachineClassifier svm,
			int[][] featuresDataset1, int[] labelsDataset1, int[][] featuresDataset2, int[] labelsDataset2) {
		double accuracy = twoFoldValidation.runTwoFoldValidation(svm, featuresDataset1, labelsDataset1,
				featuresDataset2, labelsDataset2);
		System.out.printf("SVM Overall Accuracy (Two-Fold Test): %.2f%%\n", accuracy);
	}

	// Runs two-fold validation for Multi-Layer Perceptron (MLP) classifier and
	// prints its accuracy.
	private void evaluateMLP(TwoFoldValidation twoFoldValidation, MultiLayerPerceptronClassifier mlp,
			int[][] featuresDataset1, int[] labelsDataset1, int[][] featuresDataset2, int[] labelsDataset2) {
		double accuracy = twoFoldValidation.runTwoFoldValidation(mlp, featuresDataset1, labelsDataset1,
				featuresDataset2, labelsDataset2);
		System.out.printf("MLP Overall Accuracy (Two-Fold Test): %.2f%%\n", accuracy);
	}

	// Evaluates the unified classifier combining predictions from SVM and MLP and
	// prints its accuracy.
	private void evaluateUnifiedClassifier(ClassifierEvaluator evaluator, SupportVectorMachineClassifier svm,
			MultiLayerPerceptronClassifier mlp, int[][] featuresDataset1, int[] labelsDataset1,
			int[][] featuresDataset2, int[] labelsDataset2) {
		// Evaluate accuracy for Fold 1
		double accuracyFold1 = evaluator.evaluateUnifiedClassifier(featuresDataset1, labelsDataset1, featuresDataset2,
				labelsDataset2, svm, mlp);

		// Evaluate accuracy for Fold 2
		double accuracyFold2 = evaluator.evaluateUnifiedClassifier(featuresDataset2, labelsDataset2, featuresDataset1,
				labelsDataset1, svm, mlp);

		// Print unified classifier results
		System.out.printf("Unified Classifier Accuracy (Fold 1): %.2f%%\n", accuracyFold1);
		System.out.printf("Unified Classifier Accuracy (Fold 2): %.2f%%\n", accuracyFold2);
		System.out.printf("Unified Classifier Overall Accuracy (Two-Fold Test): %.2f%%\n",
				(accuracyFold1 + accuracyFold2) / 2);
	}
}
