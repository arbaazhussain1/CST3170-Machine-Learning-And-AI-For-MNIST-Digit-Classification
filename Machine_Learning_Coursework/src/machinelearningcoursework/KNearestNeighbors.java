package machinelearningcoursework;


/**
 * Implements the k-Nearest Neighbors (kNN) classification algorithm. This
 * classifier predicts labels for testing data by finding the nearest neighbor
 * in the training data based on Euclidean distance.
 */
public class KNearestNeighbors {

   
    public KNearestNeighbors() {
        // Default constructor
    }

 // Predicts labels for the testing feature set by identifying the k-nearest neighbors from the training set based on Euclidean distance.
    public int[] predict(int[][] trainingFeatures, int[] trainingLabels, int[][] testingFeatures, int numNeighbors) {
        int[] predictedLabels = new int[testingFeatures.length]; // Store predictions for testing features
        int maxLabel = findMaxLabelValue(trainingLabels); // Determine the maximum label value for sizing arrays

        for (int testPointIndex = 0; testPointIndex < testingFeatures.length; testPointIndex++) {
            Neighbor[] nearestNeighbors = findNearestNeighbors(
                    trainingFeatures, trainingLabels, testingFeatures[testPointIndex]);
            sortNeighborsByDistance(nearestNeighbors); // Sort the neighbours by distance
            int[] labelCounts = countLabels(nearestNeighbors, numNeighbors, maxLabel); // Count labels among k-nearest neighbors
            predictedLabels[testPointIndex] = predictLabel(labelCounts); // Predict the label with majority vote
        }

        return predictedLabels; // Return the list of predictions
    }

 // Determines the maximum label value from the training labels array for classification purposes.
    private int findMaxLabelValue(int[] trainingLabels) {
        int maximumLabel = 0;  // Variable to track the highest label value
        for (int label : trainingLabels) { 
            if (label > maximumLabel) { // Update if the current label is greater than the tracked maximum
                maximumLabel = label;
            }
        }
        return maximumLabel; // Return the highest label value
    }

 // Finds the nearest neighbors for a given test feature vector by calculating Euclidean distances.
    private Neighbor[] findNearestNeighbors(int[][] trainingFeatures, int[] trainingLabels, int[] testingDataPoint) {
        Neighbor[] nearestNeighbors = new Neighbor[trainingFeatures.length]; // Array to store neighbours
        for (int trainingDataPointIndex = 0; trainingDataPointIndex < trainingFeatures.length; trainingDataPointIndex++) {
            double distance = calculateEuclideanDistance(
                    testingDataPoint, trainingFeatures[trainingDataPointIndex]); // Compute Euclidean distance
            nearestNeighbors[trainingDataPointIndex] = new Neighbor(trainingLabels[trainingDataPointIndex], distance); // Store label and distance
        }
        return nearestNeighbors; // Return the array of neighbours with distances
    }

 // Sorts an array of neighbours in ascending order of their distances using a simple bubble sort algorithm.
    private void sortNeighborsByDistance(Neighbor[] neighbors) {
        for (int currentIndex = 0; currentIndex < neighbors.length - 1; currentIndex++) {
            for (int nextIndex = currentIndex + 1; nextIndex < neighbors.length; nextIndex++) {
                if (neighbors[currentIndex].distance > neighbors[nextIndex].distance) { // Swap if out of order
                    Neighbor temporaryNeighbor = neighbors[currentIndex]; // Temporarily hold the current neighbour
                    neighbors[currentIndex] = neighbors[nextIndex];  // Swap neighbours
                    neighbors[nextIndex] = temporaryNeighbor; // Complete the swap
                }
            }
        }
    }

    // Counts the occurrences of labels among the k-nearest neighbors.
    private int[] countLabels(Neighbor[] nearestNeighbors, int numNeighbors, int maxLabel) {
        int[] labelCounts = new int[maxLabel + 1]; // Array to count occurrences of each label (size = max label + 1)
        for (int neighborIndex = 0; neighborIndex < numNeighbors; neighborIndex++) {
            int label = nearestNeighbors[neighborIndex].label; // Extract the label of the current neighbour
            labelCounts[label]++; // Increment the count for this label
        }
        return labelCounts; // Return the array containing label counts
    }


    // Predicts the label with the majority vote from the label counts array.
    private int predictLabel(int[] labelCounts) {
        int predictedLabel = 0; // Variable to store the label with the highest vote
        int maxCount = 0; // Variable to track the highest count encountered so far
        for (int label = 0; label < labelCounts.length; label++) { 
            if (labelCounts[label] > maxCount) { // Check if the current label's count exceeds the max count
                maxCount = labelCounts[label]; // Update the maximum count
                predictedLabel = label; // Update the predicted label
            }
        }
        return predictedLabel; // Return the label with the highest count (majority vote)
    }


    // Calculates the Euclidean distance between two feature vectors by summing the squared differences and taking the square root.
    private double calculateEuclideanDistance(int[] vector1, int[] vector2) {
        double squaredDifferenceSum = 0.0; // Initialise the sum of squared differences to zero

        // Iterate through each feature and compute the squared difference
        for (int featureIndex = 0; featureIndex < vector1.length; featureIndex++) {
            squaredDifferenceSum += Math.pow(vector1[featureIndex] - vector2[featureIndex], 2);
        }

        // Return the square root of the sum to get the Euclidean distance
        return Math.sqrt(squaredDifferenceSum);
    }

    // Represents a neighbouring data point with its label and distance from the test data point.
    private static class Neighbor {
        int label; // The label of the neighbour
        double distance; // The distance of the neighbour from a test point

        // Constructor to initialise the label and distance of a neighbour
        Neighbor(int label, double distance) {
            this.label = label;
            this.distance = distance;
        }
    }

}