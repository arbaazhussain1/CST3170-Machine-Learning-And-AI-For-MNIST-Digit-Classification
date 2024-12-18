package machinelearningcoursework;

import java.util.Random;

/**
 * Implements a Multi-Layer Perceptron (MLP) Classifier for multi-class
 * classification. Supports a single hidden layer with configurable size,
 * learning rate, and training iterations.
 */
public class MultiLayerPerceptronClassifier {

	// Neural network parameters
	private double[][] inputToHiddenWeights; // Weights between input and hidden layer
	private double[][] hiddenToOutputWeights; // Weights between hidden and output layer
	private double[] hiddenLayerBiases; // Biases for hidden layer neurons
	private double[] outputLayerBiases; // Biases for output layer neurons

	// Hyperparameters for training
	private double learningRate; // Learning rate for gradient updates
	private int inputFeatureCount; // Number of input features
	private int hiddenNeuronCount; // Number of neurons in the hidden layer
	private int outputNeuronCount; // Number of output neurons (classes)
	private int maxTrainingEpochs; // Maximum number of epochs for training

	// Constructor that initialises the Multi-Layer Perceptron (MLP) with random
	// weights and biases.
	public MultiLayerPerceptronClassifier(int inputFeatureCount, int hiddenNeuronCount, int outputNeuronCount,
			double learningRate, int maxTrainingEpochs) {

		// Set the dimensions for input, hidden, and output layers
		this.inputFeatureCount = inputFeatureCount;
		this.hiddenNeuronCount = hiddenNeuronCount;
		this.outputNeuronCount = outputNeuronCount;

		// Set training hyperparameters
		this.learningRate = learningRate;
		this.maxTrainingEpochs = maxTrainingEpochs;

		initialiseWeightsAndBiases(); // Assign initial random values to weights and biases
	}

	// Assigns small random values to all weights and biases.
	private void initialiseWeightsAndBiases() {
		// Create a Random instance to generate random values
		Random randomGenerator = new Random();
		// Allocate memory for input-to-hidden weight matrix
		inputToHiddenWeights = new double[inputFeatureCount][hiddenNeuronCount];
		// Allocate memory for hidden-to-output weight matrix
		hiddenToOutputWeights = new double[hiddenNeuronCount][outputNeuronCount];
		// Allocate memory for biases in the hidden layer
		hiddenLayerBiases = new double[hiddenNeuronCount];
		// Allocate memory for biases in the output layer
		outputLayerBiases = new double[outputNeuronCount];

		// Assign random values to the input-to-hidden weight matrix
		for (int featureIndex = 0; featureIndex < inputFeatureCount; featureIndex++) {
			for (int hiddenNeuronIndex = 0; hiddenNeuronIndex < hiddenNeuronCount; hiddenNeuronIndex++) {
				// Generate random weight values in the range (-0.05, 0.05)
				inputToHiddenWeights[featureIndex][hiddenNeuronIndex] = (randomGenerator.nextDouble() - 0.5) / 10;
			}
		}
		// Assign random values to the hidden-to-output weight matrix and hidden layer
		// biases
		for (int hiddenNeuronIndex = 0; hiddenNeuronIndex < hiddenNeuronCount; hiddenNeuronIndex++) {
			// Generate random bias values in the range (-0.05, 0.05) for the hidden layer
			hiddenLayerBiases[hiddenNeuronIndex] = (randomGenerator.nextDouble() - 0.5) / 10;
			for (int outputNeuronIndex = 0; outputNeuronIndex < outputNeuronCount; outputNeuronIndex++) {
				// Generate random weight values in the range (-0.05, 0.05)
				hiddenToOutputWeights[hiddenNeuronIndex][outputNeuronIndex] = (randomGenerator.nextDouble() - 0.5) / 10;
			}
		}
		// Assign random values to the output layer biases
		for (int outputNeuronIndex = 0; outputNeuronIndex < outputNeuronCount; outputNeuronIndex++) {
			// Generate random bias values in the range (-0.05, 0.05) for the output layer
			outputLayerBiases[outputNeuronIndex] = (randomGenerator.nextDouble() - 0.5) / 10;
		}
	}

	// Trains the MLP using backpropagation on the dataset's.
	public void train(int[][] trainingFeatures, int[] trainingLabels) {
		// Convert class labels to one-hot encoded vectors for multi-class
		// classification
		double[][] oneHotLabels = oneHotEncodeLabels(trainingLabels);

		// Iterate through epochs and train the network on all samples
		for (int epochIndex = 0; epochIndex < maxTrainingEpochs; epochIndex++) {
			// Loop through each training sample in the dataset's
			for (int sampleIndex = 0; sampleIndex < trainingFeatures.length; sampleIndex++) {
				// Train the network using a single sample and its corresponding target output
				trainSingleSample(trainingFeatures[sampleIndex], oneHotLabels[sampleIndex]);
			}
		}
	}

	// Converts class labels into one-hot encoded vectors for multi-class
	// classification.
	private double[][] oneHotEncodeLabels(int[] trainingLabels) {
		// Create a 2D array to hold the one-hot encoded labels
		double[][] oneHotEncodedLabels = new double[trainingLabels.length][outputNeuronCount];
		// Loop through each label and set the corresponding index in the one-hot vector
		// to 1.0
		for (int labelIndex = 0; labelIndex < trainingLabels.length; labelIndex++) {
			oneHotEncodedLabels[labelIndex][trainingLabels[labelIndex]] = 1.0; // Set the target class position to 1.0
		}
		// Return the 2D array containing one-hot encoded labels
		return oneHotEncodedLabels;
	}

	// Trains the network on a single input-output pair using forward and backward
	// passes.
	private void trainSingleSample(int[] inputFeatures, double[] targetOutput) {
		// Step 1: Forward pass - Compute activations for the hidden layer
		double[] hiddenLayerActivations = computeHiddenActivations(inputFeatures);
		// Step 2: Forward pass - Compute activations for the output layer
		double[] outputLayerActivations = computeOutputActivations(hiddenLayerActivations);
		// Step 3: Calculate errors for the output layer
		// The error is the difference between the expected and actual output
		// activations
		double[] outputLayerErrors = calculateOutputErrors(outputLayerActivations, targetOutput);
		// Step 4: Backpropagate the output layer errors to calculate errors for the
		// hidden layer
		// This involves propagating the errors backward through the network using the
		// weights
		double[] hiddenLayerErrors = calculateHiddenErrors(hiddenLayerActivations, outputLayerErrors);
		// Step 5: Update the weights and biases of the network based on the calculated
		// errors
		updateWeightsAndBiases(inputFeatures, hiddenLayerActivations, outputLayerErrors, hiddenLayerErrors);

	}

	// Computes the activations of the hidden layer neurons using sigmoid
	// activation.
	private double[] computeHiddenActivations(int[] inputFeatures) {
		// Array to store activation values for the hidden layer neurons
		double[] hiddenActivations = new double[hiddenNeuronCount];
		// Iterate through each neuron in the hidden layer
		for (int hiddenNeuronIndex = 0; hiddenNeuronIndex < hiddenNeuronCount; hiddenNeuronIndex++) {
			// Compute the weighted sum of inputs for the current hidden neuron and add its
			// bias
			hiddenActivations[hiddenNeuronIndex] = sigmoid(
					dotProduct(inputFeatures, inputToHiddenWeights, hiddenNeuronIndex)
							+ hiddenLayerBiases[hiddenNeuronIndex]); // Compute activation for each hidden neuron
		}
		// Return the computed activations for all hidden layer neurons
		return hiddenActivations;
	}

	// Computes the activations of the output layer neurons using softmax
	// activation.
	private double[] computeOutputActivations(double[] hiddenActivations) {
		// Compute the weighted inputs to the output layer neurons and add biases
		double[] outputNeuronInputs = addElementWise(dotProduct(hiddenActivations, hiddenToOutputWeights), // Weighted
																											// sums for
																											// output
																											// neurons
				outputLayerBiases); // Add biases to the weighted sums

		// Apply the softmax function to calculate the activations as probabilities
		return softmax(outputNeuronInputs);
	}

	// Calculates errors for the output layer by comparing activations with target
	// values.
	private double[] calculateOutputErrors(double[] outputActivations, double[] targetOutput) {
		double[] outputErrors = new double[outputNeuronCount];
		// Calculate the error for each output neuron
		for (int outputNeuronIndex = 0; outputNeuronIndex < outputNeuronCount; outputNeuronIndex++) {
			outputErrors[outputNeuronIndex] = targetOutput[outputNeuronIndex] - outputActivations[outputNeuronIndex];

		}
		return outputErrors; // Return the calculated errors for the output layer neurons
	}

	// Calculates errors for the hidden layer by backpropagating errors from the
	// output layer.
	private double[] calculateHiddenErrors(double[] hiddenActivations, double[] outputErrors) {
		double[] hiddenErrors = new double[hiddenNeuronCount];
		// Calculate the error for each hidden neuron
		for (int hiddenNeuronIndex = 0; hiddenNeuronIndex < hiddenNeuronCount; hiddenNeuronIndex++) {
			double errorSum = 0.0;
			// Sum the weighted errors from all connected output neurons
			for (int outputNeuronIndex = 0; outputNeuronIndex < outputNeuronCount; outputNeuronIndex++) {
				errorSum += outputErrors[outputNeuronIndex]
						* hiddenToOutputWeights[hiddenNeuronIndex][outputNeuronIndex];

			}
			// Scale the error sum by the derivative of the hidden neuron's activation
			hiddenErrors[hiddenNeuronIndex] = errorSum * sigmoidDerivative(hiddenActivations[hiddenNeuronIndex]);
		}
		return hiddenErrors; // Return the calculated errors for the hidden layer neurons
	}

	// Updates all weights and biases based on calculated gradients.
	private void updateWeightsAndBiases(int[] inputFeatures, double[] hiddenActivations, double[] outputErrors,
			double[] hiddenErrors) {
		updateHiddenToOutputWeights(hiddenActivations, outputErrors); // Update weights between hidden and output layers
		updateOutputBiases(outputErrors); // Update biases for output layer
		updateInputToHiddenWeights(inputFeatures, hiddenErrors); // Update weights between input and hidden layers
		updateHiddenBiases(hiddenErrors); // Update biases for hidden layer
	}

	// Updates weights between hidden and output layers using calculated errors.
	private void updateHiddenToOutputWeights(double[] hiddenActivations, double[] outputErrors) {
		// Iterate through all output neurons for the current hidden neuron
		for (int hiddenNeuronIndex = 0; hiddenNeuronIndex < hiddenNeuronCount; hiddenNeuronIndex++) {
			for (int outputNeuronIndex = 0; outputNeuronIndex < outputNeuronCount; outputNeuronIndex++) {
				// Update weight using the formula:
				// weight = weight + (learningRate * outputError * hiddenActivation)
				hiddenToOutputWeights[hiddenNeuronIndex][outputNeuronIndex] += learningRate
						* outputErrors[outputNeuronIndex] * hiddenActivations[hiddenNeuronIndex];
			}
		}
	}

	// Updates biases for the output layer using calculated errors.
	private void updateOutputBiases(double[] outputErrors) {
		for (int outputNeuronIndex = 0; outputNeuronIndex < outputNeuronCount; outputNeuronIndex++) {
			// Update the bias using the formula:
			// bias = bias + (learningRate * outputError)
			outputLayerBiases[outputNeuronIndex] += learningRate * outputErrors[outputNeuronIndex];
		}
	}

	// Updates weights between input and hidden layers using calculated errors.
	private void updateInputToHiddenWeights(int[] inputFeatures, double[] hiddenErrors) {
		for (int featureIndex = 0; featureIndex < inputFeatureCount; featureIndex++) {
			for (int hiddenNeuronIndex = 0; hiddenNeuronIndex < hiddenNeuronCount; hiddenNeuronIndex++) {
				// Update the weight connecting input feature `featureIndex` to hidden neuron
				// `hiddenNeuronIndex`
				inputToHiddenWeights[featureIndex][hiddenNeuronIndex] += learningRate * hiddenErrors[hiddenNeuronIndex]
						* inputFeatures[featureIndex];
			}
		}
	}

	// Updates biases for the hidden layer using calculated errors.
	private void updateHiddenBiases(double[] hiddenErrors) {
		for (int hiddenNeuronIndex = 0; hiddenNeuronIndex < hiddenNeuronCount; hiddenNeuronIndex++) {
			// Update the bias of hidden neuron `hiddenNeuronIndex`
			hiddenLayerBiases[hiddenNeuronIndex] += learningRate * hiddenErrors[hiddenNeuronIndex];
		}
	}

	// Predicts the class label for a given input feature vector by performing a
	// forward pass through the neural network.
	public int predict(int[] inputFeatures) {
		// Step 1: Compute hidden layer activations
		double[] hiddenActivations = computeHiddenActivations(inputFeatures);
		// Step 2: Compute output layer activations
		double[] outputActivations = computeOutputActivations(hiddenActivations);
		// Step 3: Return the index of the output neuron with the highest activation
		return findMaxActivationIndex(outputActivations);
	}

	// Finds the index of the neuron with the maximum activation in the output
	// layer.
	private int findMaxActivationIndex(double[] outputActivations) {
		// Initialise the predicted label as the first output neuron
		int predictedLabel = 0;
		// Set the initial maximum activation value
		double maxActivation = outputActivations[0];
		// Iterate through all output neurons to find the maximum activation value
		for (int outputNeuronIndex = 1; outputNeuronIndex < outputActivations.length; outputNeuronIndex++) {
			if (outputActivations[outputNeuronIndex] > maxActivation) {
				// Update max activation and predicted label
				maxActivation = outputActivations[outputNeuronIndex];
				predictedLabel = outputNeuronIndex; // Update predicted label
			}
		}
		return predictedLabel; // Return the index of the neuron with the highest activation
	}

	// Adds corresponding elements of two arrays and returns the resulting array.
	private double[] addElementWise(double[] array1, double[] array2) {
		// Initialise a result array of the same length as the input arrays.
		double[] result = new double[array1.length];
		// Iterate through each index and compute the element-wise sum.
		for (int arrayIndex = 0; arrayIndex < array1.length; arrayIndex++) {
			result[arrayIndex] = array1[arrayIndex] + array2[arrayIndex]; // Add corresponding elements
		}
		return result; // Return the resulting array containing the element-wise sums.
	}

	// Computes the dot product for a single column in the weight matrix.
	// Used to calculate the weighted input for a specific hidden neuron.
	private double dotProduct(int[] inputVector, double[][] weights, int columnIndex) {
		double sum = 0.0; // Initialise the weighted sum to zero.
		// Iterate through each feature in the input vector.
		for (int featureIndex = 0; featureIndex < inputVector.length; featureIndex++) {
			// Multiply the input feature value with the corresponding weight and accumulate
			// the result.
			sum += inputVector[featureIndex] * weights[featureIndex][columnIndex];
		}
		return sum; // Return the calculated weighted sum for the specified neuron.
	}

	// Computes the dot product for all columns in the weight matrix.
	// Used to calculate weighted inputs for all output neurons.
	private double[] dotProduct(double[] inputVector, double[][] weights) {
		// Initialise the result array to store the weighted sums for all output
		// neurons.
		double[] result = new double[weights[0].length];
		// Iterate through each output neuron (each column in the weight matrix).
		for (int outputNeuronIndex = 0; outputNeuronIndex < weights[0].length; outputNeuronIndex++) {
			// Compute the dot product for the current output neuron.
			for (int hiddenNeuronIndex = 0; hiddenNeuronIndex < inputVector.length; hiddenNeuronIndex++) {
				// Multiply the input value with the corresponding weight and accumulate the
				// result.
				result[outputNeuronIndex] += inputVector[hiddenNeuronIndex]
						* weights[hiddenNeuronIndex][outputNeuronIndex];
			}
		}
		return result; // Return the array of weighted sums for all output neurons.
	}

	// Applies the sigmoid activation function to a given input.
	private double sigmoid(double value) {
		// Compute the sigmoid function: 1 / (1 + exp(-value)).
		return 1 / (1 + Math.exp(-value)); // Return the result of the sigmoid activation
	}

	// Computes the derivative of the sigmoid function for backpropagation.
	private double sigmoidDerivative(double value) {
		// Compute the derivative of the sigmoid function: value * (1 - value).
		return value * (1 - value); // Return the result of the sigmoid derivative
	}

	// Applies softmax activation to a vector.
	private double[] softmax(double[] values) {
		// Find the maximum value in the input vector to improve numerical stability.
		double max = Double.NEGATIVE_INFINITY;
		for (double value : values) {
			if (value > max) {
				max = value; // Update max if the current value is greater
			}
		}

		// Compute exponentials of the adjusted values and calculate their sum.
		double sum = 0.0; // Initialise the sum of exponentials
		double[] expValues = new double[values.length]; // Array to store exponential values
		for (int valueIndex = 0; valueIndex < values.length; valueIndex++) {
			expValues[valueIndex] = Math.exp(values[valueIndex] - max); // Subtract max for stability
			sum += expValues[valueIndex]; // Accumulate the sum of exponentials
		}
		// Normalise each exponential value by dividing by the sum to get probabilities.
		for (int expIndex = 0; expIndex < expValues.length; expIndex++) {
			expValues[expIndex] /= sum; // Divide each exponential by the total sum
		}
		return expValues; // Return the normalised probabilities
	}
}
