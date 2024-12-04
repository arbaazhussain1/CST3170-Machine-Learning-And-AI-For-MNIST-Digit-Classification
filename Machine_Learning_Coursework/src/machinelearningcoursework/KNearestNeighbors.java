package machinelearningcoursework;

import java.util.ArrayList;
import java.util.List;

public class KNearestNeighbors {

    public KNearestNeighbors() {
        // Default constructor (no initialisation required for kNN in this case)
    }

    /**
     * Predicts the labels for a testing feature set using the k-Nearest Neighbors
     * algorithm.
     * 
     * @param trainingFeatures The training feature set.
     * @param trainingLabels   The training labels.
     * @param testingFeatures  The testing feature set.
     * @return Predicted labels for the testing feature set.
     */
    public List<Integer> predict(List<int[]> trainingFeatures, List<Integer> trainingLabels, List<int[]> testingFeatures) {
        List<Integer> predictedLabels = new ArrayList<>();
        for (int[] testingSample : testingFeatures) {
            double smallestDistance = Double.MAX_VALUE; // Initialise with a large value
            int nearestLabel = -1; // Initialise with a default value

            // Find the nearest neighbor in the training set
            for (int trainingIndex = 0; trainingIndex < trainingFeatures.size(); trainingIndex++) {
                double distance = calculateEuclideanDistance(testingSample, trainingFeatures.get(trainingIndex));
                if (distance < smallestDistance) {
                    smallestDistance = distance; // Update smallest distance
                    nearestLabel = trainingLabels.get(trainingIndex); // Update nearest label
                }
            }
            predictedLabels.add(nearestLabel); // Add the predicted label
        }
        return predictedLabels;
    }

    /**
     * Calculates the Euclidean distance between two feature vectors.
     * 
     * @param point1 First feature vector.
     * @param point2 Second feature vector.
     * @return Euclidean distance between the two vectors.
     */
    private double calculateEuclideanDistance(int[] point1, int[] point2) {
        double squaredDifferenceSum = 0.0;
        for (int featureIndex = 0; featureIndex < point1.length; featureIndex++) {
            squaredDifferenceSum += Math.pow(point1[featureIndex] - point2[featureIndex], 2);
        }
        return Math.sqrt(squaredDifferenceSum); // Return the square root of the sum
    }
}
