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
        for (int[] testPoint : testingFeatures) {
            double nearestDistance = Double.MAX_VALUE; // Initialise with a large value
            int labelOfNearestNeighbor = -1; // Initialise with a default value

            // Find the nearest neighbor in the training set
            for (int trainingPointIndex = 0; trainingPointIndex < trainingFeatures.size(); trainingPointIndex++) {
                double distance = calculateEuclideanDistance(testPoint, trainingFeatures.get(trainingPointIndex));
                if (distance < nearestDistance) {
                    nearestDistance = distance; // Update smallest distance
                    labelOfNearestNeighbor = trainingLabels.get(trainingPointIndex); // Update nearest label
                }
            }
            predictedLabels.add(labelOfNearestNeighbor); // Add the predicted label
        }
        return predictedLabels;
    }

    /**
     * Calculates the Euclidean distance between two feature vectors.
     * 
     * @param vector1 First feature vector.
     * @param vector2 Second feature vector.
     * @return Euclidean distance between the two vectors.
     */
    private double calculateEuclideanDistance(int[] vector1, int[] vector2) {
        double squaredDifferenceSum = 0.0;
        for (int featureIndex = 0; featureIndex < vector1.length; featureIndex++) {
            squaredDifferenceSum += Math.pow(vector1[featureIndex] - vector2[featureIndex], 2);
        }
        return Math.sqrt(squaredDifferenceSum); // Return the square root of the sum
    }
}
