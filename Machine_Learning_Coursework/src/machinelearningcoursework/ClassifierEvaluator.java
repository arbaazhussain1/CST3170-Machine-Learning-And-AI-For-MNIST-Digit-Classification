package machinelearningcoursework;

/**
 * Evaluates classifiers and calculates accuracy metrics. Supports individual
 * and unified evaluation of kNN, SVM, and MLP classifiers.
 */
public class ClassifierEvaluator {

	// Constants for kNN configuration
	private static final int KNN_NEIGHBORS = 3; // Number of nearest neighbors

	// Constants for SupportVectorMachineClassifier configuration
	private static final int SVM_NUM_CLASSES = 10; // Number of classes (0-9)
	private static final double SVM_LEARNING_RATE = 0.005; // Learning rate for SVM
	private static final double SVM_REGULARISATION_PARAM = 1.0; // Regularisation parameter
	private static final int SVM_MAX_TRAINING_ITERATIONS = 10000; // Maximum number of training iterations

	
	// Evaluates k-Nearest Neighbors (kNN) classifier by predicting labels for test
	// data and computing accuracy.
	public double evaluateKNN(KNearestNeighbors knnClassifier, int[][] trainingFeatures, int[] trainingLabels,
            int[][] testingFeatures, int[] testingLabels, int numNeighbors) {
		// Predict labels for the testing features using the kNN classifier
	    int[] predictedLabels = knnClassifier.predict(trainingFeatures, trainingLabels, testingFeatures, numNeighbors);

		// Calculate and return the accuracy of the kNN predictions
	    return calculateAccuracy(testingLabels, predictedLabels);
	}

	// Evaluates a unified classifier by combining predictions from kNN, SVM, and MLP classifiers into meta-features and training a meta-classifier to compute accuracy.
	public double evaluateUnifiedClassifier(int[][] trainingFeatures, int[] trainingLabels, int[][] testingFeatures,
			int[] testingLabels, SupportVectorMachineClassifier svmClassifier,
			MultiLayerPerceptronClassifier mlpClassifier) {

		// Generate predictions from the kNN classifier
		int[] knnPredictions = new KNearestNeighbors().predict(trainingFeatures, trainingLabels, testingFeatures,
				KNN_NEIGHBORS);

		// Initialise arrays for storing predictions from SVM and MLP
		int[] svmPredictions = new int[testingFeatures.length];
		int[] mlpPredictions = new int[testingFeatures.length];

		// Generate predictions from SVM and MLP for each test sample
		for (int testSample = 0; testSample < testingFeatures.length; testSample++) {
			svmPredictions[testSample] = svmClassifier.predict(testingFeatures[testSample]); // SVM prediction
			mlpPredictions[testSample] = mlpClassifier.predict(testingFeatures[testSample]); // MLP prediction
		}

		// Combine predictions from kNN, SVM, and MLP into a meta-feature set
		int numMetaFeatures = 3; // Number of meta-features (kNN, SVM, and MLP predictions)
		int[][] stackedMetaFeatures = new int[testingFeatures.length][numMetaFeatures];
		for (int dataPointIndex = 0; dataPointIndex < testingFeatures.length; dataPointIndex++) {
			stackedMetaFeatures[dataPointIndex] = new int[] { knnPredictions[dataPointIndex],
					svmPredictions[dataPointIndex], mlpPredictions[dataPointIndex] };
		}
		// Initialise a Support Vector Machine (SVM) as a meta-classifier
		SupportVectorMachineClassifier metaClassifier = new SupportVectorMachineClassifier(numMetaFeatures,
				SVM_NUM_CLASSES, SVM_LEARNING_RATE, SVM_REGULARISATION_PARAM, SVM_MAX_TRAINING_ITERATIONS);
		// Train the meta-classifier on the meta-feature set
		metaClassifier.train(stackedMetaFeatures, testingLabels);

		// Generate predictions using the trained meta-classifier
		int[] metaPredictions = new int[stackedMetaFeatures.length];
		for (int dataPointIndexPredictions = 0; dataPointIndexPredictions < stackedMetaFeatures.length; dataPointIndexPredictions++) {
			metaPredictions[dataPointIndexPredictions] = metaClassifier
					.predict(stackedMetaFeatures[dataPointIndexPredictions]);
		}
		// Calculate and return the accuracy of the unified classifier
		return calculateAccuracy(testingLabels, metaPredictions);
	}

	
	// Computes the accuracy of predicted labels against actual labels, while
	// generating and printing a confusion matrix.
	public double calculateAccuracy(int[] actualLabels, int[] predictedLabels) {
		int correctCount = 0; // Counter for correctly predicted labels
		int[][] confusionMatrix = new int[SVM_NUM_CLASSES][SVM_NUM_CLASSES]; // Confusion matrix dimensions

		// Iterate through actual and predicted labels to count matches
		for (int testingDataPointIndex = 0; testingDataPointIndex < actualLabels.length; testingDataPointIndex++) {
			if (actualLabels[testingDataPointIndex] == predictedLabels[testingDataPointIndex]) {
				correctCount++; // Increment counter if the prediction matches the actual label
			}

			confusionMatrix[actualLabels[testingDataPointIndex]][predictedLabels[testingDataPointIndex]]++;

			// Print the actual and predicted labels for debugging
//            System.out.println(actualLabels[testingDataPointIndex] + " " + predictedLabels[testingDataPointIndex]);
		}

		System.out.println("Confusion Matrix");
		for (int actualLabel = 0; actualLabel < SVM_NUM_CLASSES; actualLabel++) { // Row represents the actual label
			System.out.printf("%-3d | ", actualLabel);
			for (int predictedLabel = 0; predictedLabel < SVM_NUM_CLASSES; predictedLabel++) { // Column represents the
																								// predicted label
				System.out.printf("%-5d", confusionMatrix[actualLabel][predictedLabel]);
			}
			System.out.println();
		}

		// Return the accuracy as a percentage
		return (correctCount * 100.0) / actualLabels.length;
	}
}
