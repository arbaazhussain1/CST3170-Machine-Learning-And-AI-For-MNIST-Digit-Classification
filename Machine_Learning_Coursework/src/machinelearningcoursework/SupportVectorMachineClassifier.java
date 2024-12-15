package machinelearningcoursework;

/**
 * Implements a Support Vector Machine (SVM) classifier for multi-class
 * classification. Supports training using a one-vs-all strategy with hinge loss
 * optimisation and evaluates predictions by calculating class scores for each
 * input feature vector. Provides gradient descent for training with
 * regularisation to prevent over-fitting.
 */

public class SupportVectorMachineClassifier {

	// Support Vector Machine (SVM) parameters
	private double[][] classFeatureWeights; // Weights for each class
	private double[] classBiasTerms; // Bias terms for each class
	private double learningRate; // Learning rate
	private double regularisationParam; // Regularisation parameter
	private int maxIterations; // Maximum iterations for training
	private int numClasses; // Number of classes

	public SupportVectorMachineClassifier(int numFeatures, int numClasses, double learningRate,
			double regularisationParam, int maxIterations) {
		this.numClasses = numClasses; // Store the number of output classes
		this.classFeatureWeights = new double[numClasses][numFeatures]; // Initialise weights for each class
		this.classBiasTerms = new double[numClasses]; // Initialise biases for each class
		this.learningRate = learningRate; // Set the learning rate for training
		this.regularisationParam = regularisationParam; // Set the regularisation parameter
		this.maxIterations = maxIterations; // Set the maximum training iterations
	}

	// Trains the SVM classifier for each class in a one-vs-all approach.
	public void train(int[][] datasetFeatures, int[] datasetLabels) {
		System.out.println("Starting training...");
		// Iterate over all classes and train a separate classifier for each
		for (int classIndex = 0; classIndex < numClasses; classIndex++) {
			System.out.printf("Training classifier for class %d...\n", classIndex);
			trainOneClass(datasetFeatures, datasetLabels, classIndex); // Train for a specific class
		}
		System.out.println("Training complete!"); // Print statement of Training complete!
	}

	// Trains the SVM for a specific class using a one-vs-all approach.
	private void trainOneClass(int[][] datasetFeatures, int[] datasetLabels, int classIndex) {
		// Converts dataset labels into binary labels for the specified class
		int[] binaryLabels = encodeLabelsForClass(datasetLabels, classIndex);
		int numFeatures = classFeatureWeights[classIndex].length; // Number of features for this class

		// Perform gradient descent for the specified number of iterations
		for (int iter = 0; iter < maxIterations; iter++) {
			double[] gradients = new double[numFeatures]; // Stores computed gradients for feature weights
			double biasGradient = computeGradients(datasetFeatures, binaryLabels, classIndex, gradients); // Compute
																											// gradients

			updateWeightsAndBias(classIndex, gradients, biasGradient, datasetFeatures.length); // Update weights and
																								// bias

			// Periodically log progress of training
			if (iter % 100 == 0) {
				System.out.printf("Iteration %d completed for class %d...\n", iter, classIndex);
			}
		}
	}

	// Computes the gradients for weights and bias for a specific class during
	// training.
	private double computeGradients(int[][] datasetFeatures, int[] binaryLabels, int classIndex, double[] gradients) {
		double biasGradient = 0.0; // Initialise bias gradient
		// Iterate through all data points in the dataset
		for (int dataPointIndex = 0; dataPointIndex < datasetFeatures.length; dataPointIndex++) {
			int[] featureVector = datasetFeatures[dataPointIndex]; // Extract feature vector of the current sample
			int label = binaryLabels[dataPointIndex]; // Get binary label for the current class

			// Compute the margin for the sample using the current weights and bias
			double margin = label * (calculateDotProduct(classFeatureWeights[classIndex], featureVector)
					+ classBiasTerms[classIndex]);

			if (margin < 1) { // If the sample is misclassified or within the margin
				// Compute gradients for the weights
				for (int featureIndex = 0; featureIndex < featureVector.length; featureIndex++) { // Accumulate gradient
																									// for weight
																									// featureIndex
					gradients[featureIndex] += -label * featureVector[featureIndex];
				}
				// Accumulate gradient for the bias
				biasGradient += -label;
			}
		}
		return biasGradient; // Return the accumulated bias gradient
	}

	// Updates the weights and bias for a specific class using calculated gradients.
	private void updateWeightsAndBias(int classIndex, double[] gradients, double biasGradient, int numSamples) {
		int numFeatures = gradients.length; // Number of features in the dataset

		// Update each weight for the given class
		for (int featureIndex = 0; featureIndex < numFeatures; featureIndex++) {
			// Adjust gradient with regularisation and average over samples
			gradients[featureIndex] = (gradients[featureIndex] / numSamples)
					+ regularisationParam * classFeatureWeights[classIndex][featureIndex];
			// Update the weight using the adjusted gradient and learning rate
			classFeatureWeights[classIndex][featureIndex] -= learningRate * gradients[featureIndex];
		}

		// Update the bias term for the class
		classBiasTerms[classIndex] -= learningRate * (biasGradient / numSamples);
	}

	// Encodes dataset labels into binary labels for a specific class.
	private int[] encodeLabelsForClass(int[] datasetLabels, int classIndex) {
		int[] encodedLabels = new int[datasetLabels.length]; // Array to store encoded labels

		// Iterate through all dataset labels
		for (int labelIndex = 0; labelIndex < datasetLabels.length; labelIndex++) {
			// Assign +1 for the target class and -1 for all other classes
			encodedLabels[labelIndex] = (datasetLabels[labelIndex] == classIndex) ? 1 : -1;
		}

		return encodedLabels; // Return the encoded binary labels
	}

	// Predicts the class label for a given input feature vector.
	public int predict(int[] featureVector) {
		double maxScore = Double.NEGATIVE_INFINITY; // Initialise the maximum score with the smallest possible value
		int predictedClass = -1; // Initialise the predicted class as invalid (-1)

		// Iterate over all classes to compute scores
		for (int classIndex = 0; classIndex < numClasses; classIndex++) {
			// Compute the score for the current class using the dot product of weights and
			// the feature vector, plus the bias
			double score = calculateDotProduct(classFeatureWeights[classIndex], featureVector)
					+ classBiasTerms[classIndex];
			// Update the predicted class if the current score is higher than the maximum
			// score
			if (score > maxScore) {
				maxScore = score; // Update the maximum score
				predictedClass = classIndex; // Update the predicted class
			}
		}
		return predictedClass; // Return the class with the highest score
	}

	// Computes the dot product between a weight vector and a feature vector.
	private double calculateDotProduct(double[] weightVector, int[] featureVector) {
		double sum = 0.0; // Initialise the sum to accumulate the dot product result

		// Iterate over each element of the vectors
		for (int featureIndex = 0; featureIndex < weightVector.length; featureIndex++) {
			// Multiply the corresponding elements and add to the sum
			sum += weightVector[featureIndex] * featureVector[featureIndex];
		}
		return sum; // Return the computed dot product
	}
}
