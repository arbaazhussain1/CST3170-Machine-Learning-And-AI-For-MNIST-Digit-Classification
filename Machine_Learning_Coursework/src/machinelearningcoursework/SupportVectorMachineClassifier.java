package machinelearningcoursework;

import java.util.ArrayList;
import java.util.Arrays;

public class SupportVectorMachineClassifier {

	// Class variables
	private double[] featureWeights; // Weights for the features
	private double biasTerm; // Bias term for the decision boundary
	private double learningRate; // Learning rate for gradient descent
	private double regularisationParam; // Regularisation parameter to avoid over-fitting
	private int maxIterations; // Maximum number of iterations for training
	private double rbfGamma; // Gamma parameter for the RBF kernel (used in grid search)

	/**
	 * Constructor for Support Vector Machine Classifier. Initialises weights and
	 * parameters.
	 * 
	 * @param numFeatures         Number of features in the dataset.
	 * @param learningRate        Learning rate for weight updates.
	 * @param regularisationParam Regularisation parameter.
	 * @param maxIterations       Maximum number of training iterations.
	 */
	public SupportVectorMachineClassifier(int numFeatures, double learningRate, double regularisationParam,
			int maxIterations) {
		this.featureWeights = new double[numFeatures]; // Initialise feature weights to zero
		this.biasTerm = 0.0; // Initialise bias term to zero
		this.learningRate = learningRate;
		this.regularisationParam = regularisationParam;
		this.maxIterations = maxIterations;
		this.rbfGamma = 0.1; // Default gamma value for RBF kernel
	}

	/**
	 * Trains the SVM using gradient descent with hinge loss.
	 * 
	 * @param datasetFeatures Feature vectors for training as an ArrayList.
	 * @param datasetLabels   Corresponding labels for the training data as an
	 *                        ArrayList.
	 */
	public void train(ArrayList<int[]> datasetFeatures, ArrayList<Integer> datasetLabels) {
		int trainingDataSize = datasetFeatures.size();
		int featureCount = featureWeights.length;
		double[] gradients = new double[featureCount];

		System.out.println("Starting training...");

		for (int trainingIteration = 0; trainingIteration < maxIterations; trainingIteration++) {
			// Initialise gradients for this iteration
			Arrays.fill(gradients, 0);
			double biasGradient = 0;

			for (int trainingDataPointIndex = 0; trainingDataPointIndex < trainingDataSize; trainingDataPointIndex++) {
				int[] dataPointFeatures = datasetFeatures.get(trainingDataPointIndex);
				int dataPointLabel = datasetLabels.get(trainingDataPointIndex);

				// Compute prediction and margin
				double dotProduct = dotProduct(featureWeights, dataPointFeatures);
				double margin = dataPointLabel * (dotProduct + biasTerm);

				if (margin < 1) {
					// Misclassified or within margin, update weights and bias
					for (int weightIndex = 0; weightIndex < featureCount; weightIndex++) {
						gradients[weightIndex] += -dataPointLabel * dataPointFeatures[weightIndex];
					}
					biasGradient += -dataPointLabel;
				}
			}

			// Apply regularisation and gradient updates
			for (int weightIndex = 0; weightIndex < featureCount; weightIndex++) {
				gradients[weightIndex] = (gradients[weightIndex] / trainingDataSize) + 2 * featureWeights[weightIndex];
				featureWeights[weightIndex] -= learningRate * gradients[weightIndex];
			}

			biasTerm -= learningRate * (biasGradient / trainingDataSize);

			// Log progress every 100 iterations
			if (trainingIteration % 100 == 0) {
				System.out.printf("Iteration %d completed...\n", trainingIteration);
			}
		}

		System.out.println("Training complete! \n");
	}

	/**
	 * Predicts the label for a given feature vector.
	 * 
	 * @param featureVector The input feature vector for prediction.
	 * @return Predicted label (1 for positive, -1 for negative).
	 */
	public int predict(int[] featureVector) {
		double result = dotProduct(featureWeights, featureVector) + biasTerm;
		return result >= 0 ? 1 : -1;
	}

	/**
	 * Evaluates the SVM on a given dataset and computes accuracy.
	 * 
	 * @param featureData Feature vectors for evaluation as an ArrayList.
	 * @param labels      Corresponding true labels for the dataset as an ArrayList.
	 * @return Accuracy as a percentage.
	 */
	public double evaluate(ArrayList<int[]> featureData, ArrayList<Integer> labels) {
		int correctPredictions = 0;
		for (int dataPointIndex = 0; dataPointIndex < featureData.size(); dataPointIndex++) {
			int prediction = predict(featureData.get(dataPointIndex));
			if (prediction == labels.get(dataPointIndex)) {
				correctPredictions++;
			}
		}
		return (correctPredictions * 100.0) / featureData.size();
	}

	/**
	 * Performs a grid search to find the best hyper-parameters (regularisation and
	 * gamma).
	 * 
	 * @param trainingFeatures   Training feature set as an ArrayList.
	 * @param trainingLabels     Training labels as an ArrayList.
	 * @param validationFeatures Validation feature set as an ArrayList.
	 * @param validationLabels   Validation labels as an ArrayList.
	 */
	public void gridSearchWithSigmoidKernel(ArrayList<int[]> trainingFeatures, ArrayList<Integer> trainingLabels,
			ArrayList<int[]> validationFeatures, ArrayList<Integer> validationLabels) {
		double bestC = 1.0, bestGamma = 0.01, bestAccuracy = 0.0;
		double alpha = 0.1; // Sigmoid kernel scale parameter
		double beta = 0.0; // Sigmoid kernel offset parameter

		double[] regularisationValues = { 0.01, 0.1, 1, 10, 100 }; // Values to test for regularisation parameter
		double[] gammaValues = { 0.001, 0.01, 0.1, 1 }; // Values to test for gamma

		for (double regParam : regularisationValues) {
			for (double gammaValue : gammaValues) {
				this.regularisationParam = regParam;
				this.rbfGamma = gammaValue;

				for (int[] feature : trainingFeatures) {
					for (int[] validationFeature : validationFeatures) {
						sigmoidKernel(feature, validationFeature, alpha, beta);
					}
				}

				// Train and evaluate with the current parameters
				train(trainingFeatures, trainingLabels);
				double validationAccuracy = evaluate(validationFeatures, validationLabels);

				// Update best parameters if accuracy improves
				if (validationAccuracy > bestAccuracy) {
					bestAccuracy = validationAccuracy;
					bestC = regParam;
					bestGamma = gammaValue;
				}
			}
		}

		// Output the best parameters and accuracy
		System.out.printf("Best C: %.2f, Best Gamma: %.2f, Best Accuracy: %.2f%%\n", bestC, bestGamma, bestAccuracy);

		// Update classifier with the best parameters
		this.regularisationParam = bestC;
		this.rbfGamma = bestGamma;
	}

	// Helper methods

	/**
	 * Computes the dot product between a weight vector and a feature vector.
	 * 
	 * @param vector1 The weight vector.
	 * @param vector2 The feature vector.
	 * @return Dot product result.
	 */
	private double dotProduct(double[] vector1, int[] vector2) {
		double sum = 0.0;
		for (int featureIndex = 0; featureIndex < vector1.length; featureIndex++) {
			sum += vector1[featureIndex] * vector2[featureIndex];
		}
		return sum;
	}

	/**
	 * Computes the Radial Basis Function (RBF) kernel value between two vectors.
	 * 
	 * @param vector1 First feature vector.
	 * @param vector2 Second feature vector.
	 * @param gamma   The gamma parameter for the RBF kernel.
	 * @return Kernel value as a double.
	 */
	private double rbfKernel(int[] vector1, int[] vector2, double gamma) {
		double squaredDifferenceSum = 0.0;
		for (int featureIndex = 0; featureIndex < vector1.length; featureIndex++) {
			squaredDifferenceSum += Math.pow(vector1[featureIndex] - vector2[featureIndex], 2);
		}
		return Math.exp(-gamma * squaredDifferenceSum);
	}

	/**
	 * Computes the Sigmoid kernel value between two vectors.
	 * 
	 * @param vector1 First feature vector.
	 * @param vector2 Second feature vector.
	 * @param scale   Scale parameter for the sigmoid kernel.
	 * @param offset  Offset parameter for the sigmoid kernel.
	 * @return Kernel value as a double.
	 */
	private double sigmoidKernel(int[] vector1, int[] vector2, double scale, double offset) {
		double dotProduct = 0.0;
		for (int featureIndex = 0; featureIndex < vector1.length; featureIndex++) {
			dotProduct += vector1[featureIndex] * vector2[featureIndex];
		}
		return Math.tanh(scale * dotProduct + offset); // Sigmoid kernel formula
	}
}
