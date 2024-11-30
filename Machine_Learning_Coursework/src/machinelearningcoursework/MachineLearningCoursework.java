/**
 * This program implements a k-Nearest Neighbors (kNN) Algorithm classifier
 * with Euclidean distance as the similarity metric. The classifier
 * uses two-fold cross-validation to evaluate its performance.
 */
package machinelearningcoursework;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class MachineLearningCoursework {

    public static void main(String[] args) {

        // The Paths to datasets files
        String dataset1Path = "dataSet1.csv";
        String dataset2Path = "dataSet2.csv";

        // Lists to hold dataset features and categories
        List<int[]> trainingFeaturesDataset1 = new ArrayList<>();
        List<int[]> trainingFeaturesDataset2 = new ArrayList<>();
        List<Integer> trainingLabelsDataset1 = new ArrayList<>();
        List<Integer> trainingLabelsDataset2 = new ArrayList<>();

        // Load data from CSV files into memory
        loadFileData(dataset1Path, trainingFeaturesDataset1, trainingLabelsDataset1);
        loadFileData(dataset2Path, trainingFeaturesDataset2, trainingLabelsDataset2);

        // Perform two-fold cross-validation
        System.out.println("\nTwo-fold cross-validation:");

        // Fold 1: Train on Dataset 1, Test on Dataset 2
        double fold1Accuracy = evaluate(trainingFeaturesDataset1, trainingLabelsDataset1,
                trainingFeaturesDataset2, trainingLabelsDataset2);

        // Fold 2: Train on Dataset 2, Test on Dataset 1
        double fold2Accuracy = evaluate(trainingFeaturesDataset2, trainingLabelsDataset2,
                trainingFeaturesDataset1, trainingLabelsDataset1);

        // Average accuracy across both folds
        double averageAccuracy = (fold1Accuracy + fold2Accuracy) / 2;

        // Print results
        System.out.printf("Fold 1 Accuracy: %.2f%%\n", fold1Accuracy);
        System.out.printf("Fold 2 Accuracy: %.2f%%\n", fold2Accuracy);
        System.out.printf("Average Accuracy: %.2f%%\n", averageAccuracy);
    }

    /**
     * Loads data from a CSV file.
     *
     * @param filePath      The path to the CSV file.
     * @param featureMatrix The list to store feature data.
     * @param labelList     The list to store class labels.
     */
    private static void loadFileData(String filePath, List<int[]> featureMatrix, List<Integer> labelList) {
        try (Scanner fileScanner = new Scanner(new File(filePath))) {
            while (fileScanner.hasNextLine()) {
                String line = fileScanner.nextLine();
                String[] elements = line.split(",");

                // Ensure the line contains exactly 65 elements (64 features + 1 label)
                if (elements.length == 65) {
                    int[] featureRow = new int[64];
                    for (int featureIndex = 0; featureIndex < 64; featureIndex++) {
                        featureRow[featureIndex] = Integer.parseInt(elements[featureIndex]);
                    }
                    featureMatrix.add(featureRow); // Add features to the feature matrix
                    labelList.add(Integer.parseInt(elements[64])); // Add label to the label list
                } else {
                    System.out.println("Line skipped due to unexpected format.");
                }
            }
        } catch (FileNotFoundException e) {
            System.err.println("File not found: " + e.getMessage());
        }
    }

    /**
     * Evaluates the kNN classifier by calculating accuracy on the test dataset.
     *
     * @param trainingFeatures  Training dataset features.
     * @param trainingLabels    Training dataset labels.
     * @param testFeatures      Test dataset features.
     * @param testLabels        Test dataset labels.
     * @return Accuracy percentage.
     */
    private static double evaluate(List<int[]> trainingFeatures, List<Integer> trainingLabels,
                                   List<int[]> testFeatures, List<Integer> testLabels) {
        // Predict categories for the test dataset using the kNN algorithm
        List<Integer> predictedLabels = categorizeData(trainingFeatures, trainingLabels, testFeatures);

        // Calculate and return the accuracy
        return calculateAccuracy(testLabels, predictedLabels);
    }

    /**
     * Implements the kNN classification algorithm. Assigns categories to test data
     * based on the nearest neighbors in the training data.
     *
     * @param trainingFeatures Training dataset features.
     * @param trainingLabels   Training dataset labels.
     * @param testFeatures     Test dataset features.
     * @return Predicted categories for the test dataset.
     */
    private static List<Integer> categorizeData(List<int[]> trainingFeatures, List<Integer> trainingLabels,
                                                List<int[]> testFeatures) {
        List<Integer> predictedLabels = new ArrayList<>();

        // Iterate over each test data point
        for (int[] testPoint : testFeatures) {
            int closestNeighborIndex = -1; // Index of the closest training data point
            double smallestDistance = Double.MAX_VALUE; // Initialise minimum distance to a large value

            // Compare the test data point to all training data points
            for (int trainingPointIndex = 0; trainingPointIndex < trainingFeatures.size(); trainingPointIndex++) {
                double distance = calculateEuclideanDistance(testPoint, trainingFeatures.get(trainingPointIndex)); // Compute Euclidean distance

                // Update the closest point if a smaller distance is found
                if (distance < smallestDistance) {
                    smallestDistance = distance;
                    closestNeighborIndex = trainingPointIndex;
                }
            }

            // Assign the label of the closest training data point
            predictedLabels.add(trainingLabels.get(closestNeighborIndex));
        }

        return predictedLabels;
    }

    /**
     * Calculates the accuracy of the kNN predictions.
     *
     * @param actualLabels    True labels of the test dataset.
     * @param predictedLabels Predicted labels for the test dataset.
     * @return Accuracy percentage.
     */
    private static double calculateAccuracy(List<Integer> actualLabels, List<Integer> predictedLabels) {
        if (actualLabels.size() != predictedLabels.size()) {
            throw new IllegalArgumentException("Mismatch in size of actual and predicted labels.");
        }

        int correctPredictionCount = 0; // Counter for correct predictions
        // Compare true and predicted labels
        for (int labelIndex = 0; labelIndex < actualLabels.size(); labelIndex++) {
            if (actualLabels.get(labelIndex).equals(predictedLabels.get(labelIndex))) {
                correctPredictionCount++;
            }
        }

        // Calculate accuracy as a percentage
        return (correctPredictionCount * 100.0) / actualLabels.size();
    }

    /**
     * Calculates the Euclidean distance between two data points. Used as the
     * distance metric for the kNN algorithm.
     *
     * @param point1 First data point.
     * @param point2 Second data point.
     * @return Euclidean distance between the two points.
     */
    private static double calculateEuclideanDistance(int[] point1, int[] point2) {
        double squaredDifferenceSum = 0.0;

        // Sum the squared differences for all dimensions
        for (int featureIndex = 0; featureIndex < 64; featureIndex++) { // Comparing the first 64 elements (features)
            squaredDifferenceSum += Math.pow(point1[featureIndex] - point2[featureIndex], 2);
        }

        // Return the square root of the sum
        return Math.sqrt(squaredDifferenceSum);
    }
}
