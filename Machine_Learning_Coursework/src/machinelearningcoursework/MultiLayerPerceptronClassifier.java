package machinelearningcoursework;

import java.util.List;
import java.util.Random;

public class MultiLayerPerceptronClassifier {

	// Weights and biases for the network
	private double[][] weightsInputToHidden; // Weights between input and hidden layer
	private double[] weightsHiddenToOutput; // Weights between hidden and output layer
	private double[] hiddenLayerBiases; // Biases for hidden layer neurons
	private double outputBias; // Bias for the output neuron

	// Hyper parameters
	private double learningRate; // Learning rate for training
	private int inputFeatureSize; // Number of input features
	private int hiddenLayerSize; // Number of neurons in the hidden layer
	private int maxTrainingIterations; // Maximum number of training iterations

	/**
	 * Constructor for the Multi-Layer Perceptron Classifier. Initialises weights
	 * and biases randomly with small values.
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

		// Initialise weights and biases with small random values
		this.weightsInputToHidden = new double[inputFeatureSize][hiddenLayerSize];
		this.weightsHiddenToOutput = new double[hiddenLayerSize];
		this.hiddenLayerBiases = new double[hiddenLayerSize];
		this.outputBias = randomGenerator.nextDouble() - 0.5;

		// Populate weightsInputToHidden with small random values
		for (int inputIndex = 0; inputIndex < inputFeatureSize; inputIndex++) {
			for (int hiddenNeuronIndex = 0; hiddenNeuronIndex < hiddenLayerSize; hiddenNeuronIndex++) {
				weightsInputToHidden[inputIndex][hiddenNeuronIndex] = (randomGenerator.nextDouble() - 0.5) / 10;
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
	public void train(List<int[]> datasetFeatures, List<Integer> datasetLabels) {
		// Normalise labels to 0 (negative) and 1 (positive)
		List<Double> normalisedLabels = datasetLabels.stream().map(label -> label > 0 ? 1.0 : 0.0).toList();

		for (int trainingIteration = 0; trainingIteration < maxTrainingIterations; trainingIteration++) {
			for (int dataPointIndex = 0; dataPointIndex < datasetFeatures.size(); dataPointIndex++) {
				int[] inputFeatures = datasetFeatures.get(dataPointIndex);
				double targetOutput = normalisedLabels.get(dataPointIndex);

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
				double outputLayerError = (targetOutput - outputActivation) * sigmoidDerivative(outputActivation);
				double[] hiddenLayerErrors = new double[hiddenLayerSize];
				for (int hiddenNeuronIndex = 0; hiddenNeuronIndex < hiddenLayerSize; hiddenNeuronIndex++) {
					hiddenLayerErrors[hiddenNeuronIndex] = outputLayerError * weightsHiddenToOutput[hiddenNeuronIndex]
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
								* hiddenLayerErrors[hiddenNeuronIndex] * inputFeatures[inputFeatureIndex];
					}
				}
				for (int hiddenNeuronIndex = 0; hiddenNeuronIndex < hiddenLayerSize; hiddenNeuronIndex++) {
					hiddenLayerBiases[hiddenNeuronIndex] += learningRate * hiddenLayerErrors[hiddenNeuronIndex];
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
	public double evaluate(List<int[]> featureData, List<Integer> labels) {
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

	// Computes the dot product between an input vector and weights for a specific
	// column
	private double dotProduct(int[] inputFeatures, double[][] weights, int columnIndex) {
		double sum = 0.0;
		for (int inputIndex = 0; inputIndex < inputFeatures.length; inputIndex++) {
			sum += inputFeatures[inputIndex] * weights[inputIndex][columnIndex];
		}
		return sum;
	}

	// Computes the dot product between two vectors
	private double dotProduct(double[] inputVector, double[] weights) {
		double sum = 0.0;
		for (int featureIndex = 0; featureIndex < inputVector.length; featureIndex++) {
			sum += inputVector[featureIndex] * weights[featureIndex];
		}
		return sum;
	}

	// Sigmoid activation function
	private double sigmoid(double value) {
		return 1 / (1 + Math.exp(-value));
	}

	// Derivative of the sigmoid function
	private double sigmoidDerivative(double value) {
		return value * (1 - value);
	}
}
