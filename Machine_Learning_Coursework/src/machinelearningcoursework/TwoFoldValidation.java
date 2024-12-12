package machinelearningcoursework;

import java.util.ArrayList;

public class TwoFoldValidation {

	private final ClassifierEvaluator evaluator;

	/**
	 * Constructor for TwoFoldValidation.
	 *
	 * @param evaluator An instance of ClassifierEvaluator to perform evaluation
	 *                  tasks.
	 */
	public TwoFoldValidation(ClassifierEvaluator evaluator) {
		this.evaluator = evaluator;
	}

	/**
	 * Executes two-fold cross-validation specifically for kNN with the `k`
	 * parameter.
	 *
	 * @param knn           The kNN classifier.
	 * @param fold1Features Features for the first fold.
	 * @param fold1Labels   Labels for the first fold.
	 * @param fold2Features Features for the second fold.
	 * @param fold2Labels   Labels for the second fold.
	 * @param k             Number of nearest neighbors to consider.
	 * @return The average accuracy across both folds as a percentage.
	 */
	public double runTwoFoldValidation(KNearestNeighbors knn, ArrayList<int[]> fold1Features,
			ArrayList<Integer> fold1Labels, ArrayList<int[]> fold2Features, ArrayList<Integer> fold2Labels, int k) {
		// Fold 1: Train on Fold 1, Test on Fold 2
		double accuracyFold1 = evaluator.evaluateKNN(knn, fold1Features, fold1Labels, fold2Features, fold2Labels, k);

		// Fold 2: Train on Fold 2, Test on Fold 1
		double accuracyFold2 = evaluator.evaluateKNN(knn, fold2Features, fold2Labels, fold1Features, fold1Labels, k);

		// Print accuracy for each fold
		System.out.printf("Fold 1 Accuracy: %.2f%%\n", accuracyFold1);
		System.out.printf("Fold 2 Accuracy: %.2f%%\n", accuracyFold2);

		// Return the average accuracy across both folds
		return (accuracyFold1 + accuracyFold2) / 2.0;
	}

	/**
	 * Executes two-fold cross-validation for generic classifiers (SVM, MLP, etc.).
	 *
	 * @param classifier    The classifier to evaluate (SVM, MLP, etc.).
	 * @param fold1Features Features for the first fold.
	 * @param fold1Labels   Labels for the first fold.
	 * @param fold2Features Features for the second fold.
	 * @param fold2Labels   Labels for the second fold.
	 * @return The average accuracy across both folds as a percentage.
	 */
	public double runTwoFoldValidation(Object classifier, ArrayList<int[]> fold1Features,
			ArrayList<Integer> fold1Labels, ArrayList<int[]> fold2Features, ArrayList<Integer> fold2Labels) {
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

	/**
	 * Evaluates a classifier by training it on one fold and testing it on another.
	 * The classifier's type is determined dynamically.
	 *
	 * @param classifier       The classifier to evaluate.
	 * @param trainingFeatures Features used for training the classifier.
	 * @param trainingLabels   Labels corresponding to the training features.
	 * @param testingFeatures  Features used for testing the classifier.
	 * @param testingLabels    Labels corresponding to the testing features.
	 * @return The accuracy of the classifier on the testing dataset as a
	 *         percentage.
	 */
	private double evaluateClassifier(Object classifier, ArrayList<int[]> trainingFeatures,
			ArrayList<Integer> trainingLabels, ArrayList<int[]> testingFeatures, ArrayList<Integer> testingLabels) {
		if (classifier instanceof KNearestNeighbors) {
			// Evaluate kNN classifier (use default k if called here)
			return evaluator.evaluateKNN((KNearestNeighbors) classifier, trainingFeatures, trainingLabels,
					testingFeatures, testingLabels, 3);
		} else if (classifier instanceof SupportVectorMachineClassifier) {
			// Evaluate SVM classifier
			SupportVectorMachineClassifier svm = (SupportVectorMachineClassifier) classifier;
			svm.train(trainingFeatures, trainingLabels); // Train SVM
			return svm.evaluate(testingFeatures, testingLabels); // Evaluate SVM
		} else if (classifier instanceof MultiLayerPerceptronClassifier) {
			// Evaluate MLP classifier
			MultiLayerPerceptronClassifier mlp = (MultiLayerPerceptronClassifier) classifier;
			mlp.train(trainingFeatures, trainingLabels); // Train MLP
			return mlp.evaluate(testingFeatures, testingLabels); // Evaluate MLP
		} else {
			// If an unsupported classifier is provided, throw an exception
			throw new IllegalArgumentException("Unsupported classifier type.");
		}
	}
}
