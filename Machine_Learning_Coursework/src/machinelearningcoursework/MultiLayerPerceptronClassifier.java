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
     * Constructor for the Multi-Layer Perceptron Classifier.
     * Initialises weights and biases randomly with small values.
     * 
     * @param inputFeatureSize       Number of input features.
     * @param hiddenLayerSize        Number of neurons in the hidden layer.
     * @param outputSize             Number of output neurons (unused but required for consistency).
     * @param learningRate           Learning rate for weight updates.
     * @param maxTrainingIterations  Maximum number of training iterations.
     */
    public MultiLayerPerceptronClassifier(int inputFeatureSize, int hiddenLayerSize, int outputSize, double learningRate,
            int maxTrainingIterations) {
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
     * @param featureData Feature matrix (list of feature vectors).
     * @param labels      List of corresponding labels for training.
     */
    public void train(List<int[]> featureData, List<Integer> labels) {
        // Normalise labels to 0 (negative) and 1 (positive)
        List<Double> normalizedLabels = labels.stream().map(label -> label > 0 ? 1.0 : 0.0).toList();

        for (int iterationCount = 0; iterationCount < maxTrainingIterations; iterationCount++) {
            for (int sampleIndex = 0; sampleIndex < featureData.size(); sampleIndex++) {
                int[] inputFeatures = featureData.get(sampleIndex);
                double targetOutput = normalizedLabels.get(sampleIndex);

                // Forward pass: Compute hidden layer outputs
                double[] hiddenLayerOutputs = new double[hiddenLayerSize];
                for (int hiddenNeuronIndex = 0; hiddenNeuronIndex < hiddenLayerSize; hiddenNeuronIndex++) {
                    hiddenLayerOutputs[hiddenNeuronIndex] = sigmoid(
                            dotProduct(inputFeatures, weightsInputToHidden, hiddenNeuronIndex)
                                    + hiddenLayerBiases[hiddenNeuronIndex]);
                }

                // Compute output layer activation
                double output = sigmoid(dotProduct(hiddenLayerOutputs, weightsHiddenToOutput) + outputBias);

                // Backpropagation: Compute errors
                double outputError = (targetOutput - output) * sigmoidDerivative(output);
                double[] hiddenErrors = new double[hiddenLayerSize];
                for (int hiddenNeuronIndex = 0; hiddenNeuronIndex < hiddenLayerSize; hiddenNeuronIndex++) {
                    hiddenErrors[hiddenNeuronIndex] = outputError * weightsHiddenToOutput[hiddenNeuronIndex]
                            * sigmoidDerivative(hiddenLayerOutputs[hiddenNeuronIndex]);
                }

                // Update weights and biases
                // Update weightsHiddenToOutput and outputBias
                for (int hiddenNeuronIndex = 0; hiddenNeuronIndex < hiddenLayerSize; hiddenNeuronIndex++) {
                    weightsHiddenToOutput[hiddenNeuronIndex] += learningRate * outputError
                            * hiddenLayerOutputs[hiddenNeuronIndex];
                }
                outputBias += learningRate * outputError;

                // Update weightsInputToHidden and hiddenLayerBiases
                for (int inputIndex = 0; inputIndex < inputFeatureSize; inputIndex++) {
                    for (int hiddenNeuronIndex = 0; hiddenNeuronIndex < hiddenLayerSize; hiddenNeuronIndex++) {
                        weightsInputToHidden[inputIndex][hiddenNeuronIndex] += learningRate
                                * hiddenErrors[hiddenNeuronIndex] * inputFeatures[inputIndex];
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
    public double evaluate(List<int[]> featureData, List<Integer> labels) {
        int correctPredictions = 0;
        for (int sampleIndex = 0; sampleIndex < featureData.size(); sampleIndex++) {
            int prediction = predict(featureData.get(sampleIndex));
            int trueLabel = labels.get(sampleIndex) > 0 ? 1 : 0;
            if (prediction == trueLabel) {
                correctPredictions++;
            }
        }
        return (correctPredictions * 100.0) / featureData.size();
    }

    // Helper methods

    // Computes the dot product between an input vector and weights for a specific column
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
