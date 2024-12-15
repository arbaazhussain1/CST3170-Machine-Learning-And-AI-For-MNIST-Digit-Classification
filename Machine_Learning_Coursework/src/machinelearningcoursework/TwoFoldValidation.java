package machinelearningcoursework;

/**
 * Manages two-fold cross-validation for evaluating classifiers. Supports
 * k-Nearest Neighbors (kNN), Support Vector Machines (SVM), and Multi-Layer
 * Perceptrons (MLP) with consistent accuracy calculation and result logging for
 * both folds.
 */

public class TwoFoldValidation {

	private final ClassifierEvaluator evaluator;

	// Initialises the TwoFoldValidation class with a given evaluator
	public TwoFoldValidation(ClassifierEvaluator evaluator) {
		this.evaluator = evaluator;
	}

	// Runs two-fold cross-validation specifically for kNN with the specified number
	// of neighbors
	public double runTwoFoldValidation(KNearestNeighbors knn, int[][] fold1Features, int[] fold1Labels,
			int[][] fold2Features, int[] fold2Labels, int numNeighbors) {
		// Fold 1: Train on Fold 1, Test on Fold 2
		double accuracyFold1 = evaluator.evaluateKNN(knn, fold1Features, fold1Labels, fold2Features, fold2Labels,
				numNeighbors);

		// Fold 2: Train on Fold 2, Test on Fold 1
		double accuracyFold2 = evaluator.evaluateKNN(knn, fold2Features, fold2Labels, fold1Features, fold1Labels,
				numNeighbors);

		// Print accuracy for each fold
		System.out.printf("Fold 1 Accuracy: %.2f%%\n", accuracyFold1);
		System.out.printf("Fold 2 Accuracy: %.2f%%\n", accuracyFold2);

		// Return the average accuracy across both folds
		return (accuracyFold1 + accuracyFold2) / 2.0;
	}

	// Runs two-fold cross-validation for a generic classifier (SVM, MLP, etc.)
	public double runTwoFoldValidation(Object classifier, int[][] fold1Features, int[] fold1Labels,
			int[][] fold2Features, int[] fold2Labels) {
		// Fold 1: Train on Fold 1, Test on Fold 2
		double accuracyFold1 = evaluateClassifier(classifier, fold1Features, fold1Labels, fold2Features, fold2Labels);

		// Fold 2: Train on Fold 2, Test on Fold 1
		double accuracyFold2 = evaluateClassifier(classifier, fold2Features, fold2Labels, fold1Features, fold1Labels);

		// Print accuracy for each fold
		System.out.printf("Fold 1 Accuracy: %.2f%%\n", accuracyFold1);
		System.out.printf("Fold 2 Accuracy: %.2f%%\n", accuracyFold2);

		// Return the average accuracy across both folds
		return (accuracyFold1 + accuracyFold2) / 2.0;
	}

	// Evaluates a classifier by training it on one fold and testing it on
	// another. The classifier's type is determined dynamically.
	private double evaluateClassifier(Object classifier, int[][] trainingFeatures, int[] trainingLabels,
			int[][] testingFeatures, int[] testingLabels) {
		if (classifier instanceof KNearestNeighbors) {
			// Evaluate kNN classifier
			return evaluator.evaluateKNN((KNearestNeighbors) classifier, trainingFeatures, trainingLabels,
					testingFeatures, testingLabels, 3);
		} else if (classifier instanceof SupportVectorMachineClassifier) {
			// Evaluate SVM classifier
			SupportVectorMachineClassifier svm = (SupportVectorMachineClassifier) classifier;
			svm.train(trainingFeatures, trainingLabels); // Train SVM
			int[] predictedLabels = new int[testingFeatures.length];
			for (int testIndex = 0; testIndex < testingFeatures.length; testIndex++) {
				predictedLabels[testIndex] = svm.predict(testingFeatures[testIndex]);
			}
			// Calculate and print confusion matrix
			return evaluator.calculateAccuracy(testingLabels, predictedLabels);
		} else if (classifier instanceof MultiLayerPerceptronClassifier) {
			// Evaluate MLP classifier
			MultiLayerPerceptronClassifier mlp = (MultiLayerPerceptronClassifier) classifier;
			mlp.train(trainingFeatures, trainingLabels); // Train MLP
			int[] predictedLabels = new int[testingFeatures.length];
			for (int testIndex = 0; testIndex < testingFeatures.length; testIndex++) {
				predictedLabels[testIndex] = mlp.predict(testingFeatures[testIndex]);
			}
			// Calculate and print confusion matrix
			return evaluator.calculateAccuracy(testingLabels, predictedLabels);
		} else {
			throw new IllegalArgumentException("Unsupported classifier type.");
		}
	}
}
