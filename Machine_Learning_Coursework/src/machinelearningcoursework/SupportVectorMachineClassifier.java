package machinelearningcoursework;

import java.util.List;

public class SupportVectorMachineClassifier {

    // Class variables
    private double[] featureWeights; // Weights for the features
    private double biasTerm; // Bias term for the decision boundary
    private double learningRate; // Learning rate for gradient descent
    private double regularisationParam; // Regularisation parameter to avoid over-fitting
    private int maxIterations; // Maximum number of iterations for training
    private double rbfGamma; // Gamma parameter for the RBF kernel (used in grid search)

    /**
     * Constructor for Support Vector Machine Classifier.
     * Initialises weights and parameters.
     * 
     * @param numFeatures       Number of features in the dataset.
     * @param learningRate      Learning rate for weight updates.
     * @param regularisationParam Regularisation parameter.
     * @param maxIterations     Maximum number of training iterations.
     */
    public SupportVectorMachineClassifier(int numFeatures, double learningRate, double regularisationParam, int maxIterations) {
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
     * @param featureData List of feature vectors for training.
     * @param labels      Corresponding labels for the training data.
     */
    public void train(List<int[]> featureData, List<Integer> labels) {
        int numSamples = featureData.size();

        for (int iterationCount = 0; iterationCount < maxIterations; iterationCount++) {
            for (int sampleIndex = 0; sampleIndex < numSamples; sampleIndex++) {
                int[] featureVector = featureData.get(sampleIndex);
                int trueLabel = labels.get(sampleIndex);

                // Compute the margin (distance from the hyperplane)
                double margin = trueLabel * (dotProduct(featureWeights, featureVector) + biasTerm);

                if (margin >= 1) {
                    // Correctly classified, only regularise weights
                    for (int weightIndex = 0; weightIndex < featureWeights.length; weightIndex++) {
                        featureWeights[weightIndex] -= learningRate * 2 * featureWeights[weightIndex];
                    }
                } else {
                    // Misclassified, update weights and bias
                    for (int weightIndex = 0; weightIndex < featureWeights.length; weightIndex++) {
                        featureWeights[weightIndex] -= learningRate * 
                                (2 * featureWeights[weightIndex] - regularisationParam * trueLabel * featureVector[weightIndex]);
                    }
                    biasTerm += learningRate * regularisationParam * trueLabel;
                }
            }
        }
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
     * @param featureData List of feature vectors for evaluation.
     * @param labels      Corresponding true labels for the dataset.
     * @return Accuracy as a percentage.
     */
    public double evaluate(List<int[]> featureData, List<Integer> labels) {
        int correctPredictions = 0;
        for (int sampleIndex = 0; sampleIndex < featureData.size(); sampleIndex++) {
            int prediction = predict(featureData.get(sampleIndex));
            if (prediction == labels.get(sampleIndex)) {
                correctPredictions++;
            }
        }
        return (correctPredictions * 100.0) / featureData.size();
    }

    /**
     * Performs a grid search to find the best hyper-parameters (regularisation and gamma).
     * 
     * @param trainingFeatures Training feature set.
     * @param trainingLabels   Training labels.
     * @param validationFeatures Validation feature set.
     * @param validationLabels Validation labels.
     */
    public void gridSearch(List<int[]> trainingFeatures, List<Integer> trainingLabels, List<int[]> validationFeatures,
            List<Integer> validationLabels) {
        double bestRegularizationParam = 1.0, bestRbfGamma = 0.1, bestAccuracy = 0.0;

        double[] regularizationValues = { 0.1, 1, 10 }; // Values to test for regularisation parameter
        double[] gammaValues = { 0.01, 0.1, 1 }; // Values to test for gamma

        for (double regParam : regularizationValues) {
            for (double gammaValue : gammaValues) {
                this.regularisationParam = regParam;
                this.rbfGamma = gammaValue;

                // Train and evaluate with the current parameters
                train(trainingFeatures, trainingLabels);
                double validationAccuracy = evaluate(validationFeatures, validationLabels);

                // Update best parameters if accuracy improves
                if (validationAccuracy > bestAccuracy) {
                    bestAccuracy = validationAccuracy;
                    bestRegularizationParam = regParam;
                    bestRbfGamma = gammaValue;
                }
            }
        }

        // Output the best parameters and accuracy
        System.out.printf("Best Regularisation Param: %.2f, Best RBF Gamma: %.2f, Best Accuracy: %.2f%%\n",
                bestRegularizationParam, bestRbfGamma, bestAccuracy);

        // Update classifier with the best parameters
        this.regularisationParam = bestRegularizationParam;
        this.rbfGamma = bestRbfGamma;
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
     * @param x1    First feature vector.
     * @param x2    Second feature vector.
     * @param gamma The gamma parameter for the RBF kernel.
     * @return Kernel value as a double.
     */
    private double rbfKernel(int[] x1, int[] x2, double gamma) {
        double sum = 0.0;
        for (int i = 0; i < x1.length; i++) {
            sum += Math.pow(x1[i] - x2[i], 2); // Compute squared difference
        }
        return Math.exp(-gamma * sum); // Return exponential of the scaled sum
    }
}
