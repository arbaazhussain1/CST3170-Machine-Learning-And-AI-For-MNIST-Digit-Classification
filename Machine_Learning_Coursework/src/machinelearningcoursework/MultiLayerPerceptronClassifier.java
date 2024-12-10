package machinelearningcoursework;

import java.util.ArrayList;
import java.util.Random;

/**
 * Implements a Multi-Layer Perceptron (MLP) Classifier for binary
 * classification. Supports a single hidden layer with configurable size,
 * learning rate, and training iterations.
 */
public class MultiLayerPerceptronClassifier {

	// Neural network parameters
	private double[][] weightsInputToHidden; // Weights between the input layer and hidden layer
	private double[] weightsHiddenToOutput; // Weights between the hidden layer and output layer
	private double[] hiddenLayerBiases; // Biases for hidden layer neurons
	private double outputBias; // Bias for the output neuron

	// Hyperparameters for training
	private double learningRate; // Learning rate for gradient updates
	private int inputFeatureSize; // Number of input features
	private int hiddenLayerSize; // Number of neurons in the hidden layer
	private int maxTrainingIterations; // Maximum number of iterations for training

	/**
	 * Constructor for the Multi-Layer Perceptron Classifier. Initialises weights
	 * and biases with small random values.
	 *
	 * @param inputFeatureSize      Number of input features.
	 * @param hiddenLayerSize       Number of neurons in the hidden layer.
	 * @param outputSize            Number of output neurons (unused but required
	 *                              for consistency).
	 * @param learningRate          Learning rate for weight updates.
	 * @param maxTrainingIterations Maximum number of training iterations.
	 */
	public MultiLayerPerceptronClassifier(int inputFeatureSize, int hiddenLayerSize, int outputSize,
			double learningRate, int maxTrainingIterations) {
		this.inputFeatureSize = inputFeatureSize;
		this.hiddenLayerSize = hiddenLayerSize;
		this.learningRate = learningRate;
		this.maxTrainingIterations = maxTrainingIterations;

		Random randomGenerator = new Random();

		// Initialise weights and biases
		weightsInputToHidden = new double[inputFeatureSize][hiddenLayerSize];
		weightsHiddenToOutput = new double[hiddenLayerSize];
		hiddenLayerBiases = new double[hiddenLayerSize];
		outputBias = randomGenerator.nextDouble() - 0.5;

		// Populate weightsInputToHidden with small random values
		for (int inputIndex = 0; inputIndex < inputFeatureSize; inputIndex++) {
			for (int j = 0; j < hiddenLayerSize; j++) {
				weightsInputToHidden[inputIndex][j] = (randomGenerator.nextDouble() - 0.5) / 10;
			}
		}

		// Populate weightsHiddenToOutput and hiddenLayerBiases with small random values
		for (int hiddenNeuronIndex = 0; hiddenNeuronIndex < hiddenLayerSize; hiddenNeuronIndex++) {
			weightsHiddenToOutput[hiddenNeuronIndex] = (randomGenerator.nextDouble() - 0.5) / 10;
			hiddenLayerBiases[hiddenNeuronIndex] = (randomGenerator.nextDouble() - 0.5) / 10;
		}
	}

	/**
	 * Trains the MLP using backpropagation on the given feature data and labels.
	 *
	 * @param datasetFeatures Feature matrix (list of feature vectors).
	 * @param datasetLabels   List of corresponding labels for training.
	 */
	public void train(ArrayList<int[]> datasetFeatures, ArrayList<Integer> datasetLabels) {
		// Normalise labels to binary format: 0 (negative) and 1 (positive)
		ArrayList<Double> normalisedLabels = new ArrayList<>();
		for (Integer label : datasetLabels) {
			normalisedLabels.add(label > 0 ? 1.0 : 0.0);
		}

		// Perform training for the specified number of iterations
		for (int trainingIteration = 0; trainingIteration < maxTrainingIterations; trainingIteration++) {
			for (int trainingDataPointIndex = 0; trainingDataPointIndex < datasetFeatures
					.size(); trainingDataPointIndex++) {
				int[] inputFeatures = datasetFeatures.get(trainingDataPointIndex); // Input feature vector
				double targetOutput = normalisedLabels.get(trainingDataPointIndex); // Target label

				// Forward pass: Compute hidden layer outputs
				double[] hiddenLayerActivations = new double[hiddenLayerSize];
				for (int hiddenLayerNeuronIndex = 0; hiddenLayerNeuronIndex < hiddenLayerSize; hiddenLayerNeuronIndex++) {
					hiddenLayerActivations[hiddenLayerNeuronIndex] = sigmoid(
							dotProduct(inputFeatures, weightsInputToHidden, hiddenLayerNeuronIndex)
									+ hiddenLayerBiases[hiddenLayerNeuronIndex]);
				}

				// Compute output layer activation
				double outputActivation = sigmoid(
						dotProduct(hiddenLayerActivations, weightsHiddenToOutput) + outputBias);

				// Backpropagation: Compute errors
				double outputLayerError = (targetOutput - outputActivation) * sigmoidDerivative(outputActivation); // Output
																													// layer
																													// error
				double[] hiddenErrors = new double[hiddenLayerSize]; // Hidden layer errors
				for (int hiddenNeuronIndex = 0; hiddenNeuronIndex < hiddenLayerSize; hiddenNeuronIndex++) {
					hiddenErrors[hiddenNeuronIndex] = outputLayerError * weightsHiddenToOutput[hiddenNeuronIndex]
							* sigmoidDerivative(hiddenLayerActivations[hiddenNeuronIndex]);
				}

				// Update weights and biases
				// Update weightsHiddenToOutput and outputBias
				for (int hiddenNeuronIndex = 0; hiddenNeuronIndex < hiddenLayerSize; hiddenNeuronIndex++) {
					weightsHiddenToOutput[hiddenNeuronIndex] += learningRate * outputLayerError
							* hiddenLayerActivations[hiddenNeuronIndex];
				}
				outputBias += learningRate * outputLayerError;

				// Update weightsInputToHidden and hiddenLayerBiases
				for (int inputFeatureIndex = 0; inputFeatureIndex < inputFeatureSize; inputFeatureIndex++) {
					for (int hiddenNeuronIndex = 0; hiddenNeuronIndex < hiddenLayerSize; hiddenNeuronIndex++) {
						weightsInputToHidden[inputFeatureIndex][hiddenNeuronIndex] += learningRate
								* hiddenErrors[hiddenNeuronIndex] * inputFeatures[inputFeatureIndex];
					}
				}
				for (int hiddenNeuronIndex = 0; hiddenNeuronIndex < hiddenLayerSize; hiddenNeuronIndex++) {
					hiddenLayerBiases[hiddenNeuronIndex] += learningRate * hiddenErrors[hiddenNeuronIndex];
				}
			}
		}
	}

	/**
	 * Predicts the label for a given input feature vector.
	 *
	 * @param inputFeatures Feature vector for prediction.
	 * @return Predicted label (1 for positive, 0 for negative).
	 */
	public int predict(int[] inputFeatures) {
		// Forward pass through the hidden layer
		double[] hiddenLayerOutputs = new double[hiddenLayerSize];
		for (int hiddenNeuronIndex = 0; hiddenNeuronIndex < hiddenLayerSize; hiddenNeuronIndex++) {
			hiddenLayerOutputs[hiddenNeuronIndex] = sigmoid(
					dotProduct(inputFeatures, weightsInputToHidden, hiddenNeuronIndex)
							+ hiddenLayerBiases[hiddenNeuronIndex]);
		}

		// Compute final output
		double output = sigmoid(dotProduct(hiddenLayerOutputs, weightsHiddenToOutput) + outputBias);

		// Return predicted label
		return output >= 0.5 ? 1 : 0;
	}

	/**
	 * Evaluates the model on a given dataset and computes accuracy.
	 *
	 * @param featureData Feature matrix (list of feature vectors).
	 * @param labels      List of true labels.
	 * @return Accuracy as a percentage.
	 */
	public double evaluate(ArrayList<int[]> featureData, ArrayList<Integer> labels) {
		int correctPredictions = 0;
		for (int dataPointIndex = 0; dataPointIndex < featureData.size(); dataPointIndex++) {
			int prediction = predict(featureData.get(dataPointIndex));
			int trueLabel = labels.get(dataPointIndex) > 0 ? 1 : 0;
			if (prediction == trueLabel) {
				correctPredictions++;
			}
		}
		return (correctPredictions * 100.0) / featureData.size();
	}

	// Helper methods

	/**
	 * Computes the dot product between an input vector and a column of weights.
	 *
	 * @param inputFeatures The input feature vector.
	 * @param weights       The weight matrix.
	 * @param columnIndex   The index of the column to compute the dot product with.
	 * @return The computed dot product.
	 */
	private double dotProduct(int[] inputFeatures, double[][] weights, int columnIndex) {
		double sum = 0.0;
		for (int inputIndex = 0; inputIndex < inputFeatures.length; inputIndex++) {
			sum += inputFeatures[inputIndex] * weights[inputIndex][columnIndex];
		}
		return sum;
	}

	/**
	 * Computes the dot product between two vectors.
	 *
	 * @param inputVector The input vector.
	 * @param weights     The weight vector.
	 * @return The computed dot product.
	 */
	private double dotProduct(double[] inputVector, double[] weights) {
		double sum = 0.0;
		for (int featureIndex = 0; featureIndex < inputVector.length; featureIndex++) {
			sum += inputVector[featureIndex] * weights[featureIndex];
		}
		return sum;
	}

	/**
	 * Computes the sigmoid activation function.
	 *
	 * @param value The input value.
	 * @return The sigmoid activation.
	 */
	private double sigmoid(double value) {
		return 1 / (1 + Math.exp(-value));
	}

	/**
	 * Computes the derivative of the sigmoid function.
	 *
	 * @param value The input value.
	 * @return The sigmoid derivative.
	 */
	private double sigmoidDerivative(double value) {
		return value * (1 - value);
	}
}
