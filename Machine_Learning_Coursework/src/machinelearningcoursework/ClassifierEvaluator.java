package machinelearningcoursework;

import java.util.ArrayList;

/**
 * Evaluates classifiers and calculates accuracy metrics. Supports individual
 * and unified evaluation of kNN, SVM, and MLP classifiers.
 */
public class ClassifierEvaluator {

	/**
	 * Evaluates the k-Nearest Neighbors (kNN) classifier.
	 *
	 * @param knnClassifier    The kNN classifier instance.
	 * @param trainingFeatures The training feature set as an ArrayList.
	 * @param trainingLabels   The training label set as an ArrayList.
	 * @param testingFeatures  The testing feature set as an ArrayList.
	 * @param testingLabels    The testing label set as an ArrayList.
	 * @return Accuracy of the kNN classifier as a percentage.
	 */
	public double evaluateKNN(KNearestNeighbors knnClassifier, ArrayList<int[]> trainingFeatures,
			ArrayList<Integer> trainingLabels, ArrayList<int[]> testingFeatures, ArrayList<Integer> testingLabels) {
		// Predict labels for the testing features using the kNN classifier
		ArrayList<Integer> predictedLabels = knnClassifier.predict(trainingFeatures, trainingLabels, testingFeatures);

		// Calculate and return the accuracy of the kNN predictions
		return calculateAccuracy(testingLabels, predictedLabels);
	}

	/**
	 * Evaluates a unified classifier using stacked predictions from kNN, SVM, and
	 * MLP.
	 *
	 * @param trainingFeatures The training feature set as an ArrayList.
	 * @param trainingLabels   The training label set as an ArrayList.
	 * @param testingFeatures  The testing feature set as an ArrayList.
	 * @param testingLabels    The testing label set as an ArrayList.
	 * @param svmClassifier    The Support Vector Machine (SVM) classifier.
	 * @param mlpClassifier    The Multi-Layer Perceptron (MLP) classifier.
	 * @return Accuracy of the unified classifier as a percentage.
	 */
	public double evaluateUnifiedClassifier(ArrayList<int[]> trainingFeatures, ArrayList<Integer> trainingLabels,
			ArrayList<int[]> testingFeatures, ArrayList<Integer> testingLabels,
			SupportVectorMachineClassifier svmClassifier, MultiLayerPerceptronClassifier mlpClassifier) {
		// Generate predictions from the kNN classifier
		ArrayList<Integer> knnPredictions = new KNearestNeighbors().predict(trainingFeatures, trainingLabels,
				testingFeatures);

		// Initialise arrays for storing predictions from SVM and MLP
		ArrayList<Integer> svmPredictions = new ArrayList<>();
		ArrayList<Integer> mlpPredictions = new ArrayList<>();

		// Generate predictions from SVM and MLP for each test sample
		for (int[] testSample : testingFeatures) {
			svmPredictions.add(svmClassifier.predict(testSample)); // SVM prediction
			mlpPredictions.add(mlpClassifier.predict(testSample)); // MLP prediction
		}

		// Combine predictions from kNN, SVM, and MLP into a meta-feature set
		ArrayList<int[]> stackedMetaFeatures = new ArrayList<>();
		for (int dataPointIndex = 0; dataPointIndex < testingFeatures.size(); dataPointIndex++) {
			stackedMetaFeatures.add(new int[] { knnPredictions.get(dataPointIndex), // kNN prediction
					svmPredictions.get(dataPointIndex), // SVM prediction
					mlpPredictions.get(dataPointIndex) // MLP prediction
			});
		}

		// Initialise a Support Vector Machine (SVM) as a meta-classifier
		SupportVectorMachineClassifier metaClassifier = new SupportVectorMachineClassifier(3, 0.01, 1.0, 1000);

		// Train the meta-classifier on the meta-feature set
		metaClassifier.train(stackedMetaFeatures, testingLabels);

		// Generate predictions using the trained meta-classifier
		ArrayList<Integer> metaPredictions = new ArrayList<>();
		for (int[] metaFeature : stackedMetaFeatures) {
			metaPredictions.add(metaClassifier.predict(metaFeature));
		}

		// Calculate and return the accuracy of the unified classifier
		return calculateAccuracy(testingLabels, metaPredictions);
	}

	/**
	 * Calculates the accuracy of predictions compared to actual labels.
	 *
	 * @param actualLabels    The true labels as an ArrayList.
	 * @param predictedLabels The predicted labels as an ArrayList.
	 * @return Accuracy as a percentage.
	 */
	public double calculateAccuracy(ArrayList<Integer> actualLabels, ArrayList<Integer> predictedLabels) {
		int correctCount = 0; // Counter for correctly predicted labels

		// Iterate through actual and predicted labels to count matches
		for (int testingDataPointIndex = 0; testingDataPointIndex < actualLabels.size(); testingDataPointIndex++) {
			if (actualLabels.get(testingDataPointIndex).equals(predictedLabels.get(testingDataPointIndex))) {
				correctCount++; // Increment counter if the prediction matches the actual label
			}
		}

		// Return the accuracy as a percentage
		return (correctCount * 100.0) / actualLabels.size();
	}
}
