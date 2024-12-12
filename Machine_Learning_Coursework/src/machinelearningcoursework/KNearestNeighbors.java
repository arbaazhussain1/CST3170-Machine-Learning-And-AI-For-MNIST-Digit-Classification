package machinelearningcoursework;

import java.util.ArrayList;

/**
 * Implements the k-Nearest Neighbors (kNN) classification algorithm. This
 * classifier predicts labels for testing data by finding the nearest neighbor
 * in the training data based on Euclidean distance.
 */
public class KNearestNeighbors {

	/**
	 * Default constructor for the K-Nearest Neighbors (kNN) classifier. No
	 * initialisation required for this implementation.
	 */
	public KNearestNeighbors() {
		// Default constructor
	}

	/**
	 * Predicts the labels for a testing feature set using the k-Nearest Neighbors
	 * algorithm.
	 *
	 * @param trainingFeatures The training feature set as an ArrayList of feature
	 *                         vectors.
	 * @param trainingLabels   The corresponding labels for the training feature
	 *                         set.
	 * @param testingFeatures  The testing feature set as an ArrayList of feature
	 *                         vectors.
	 * @param k                Number of nearest neighbors to consider.
	 * @return An ArrayList of predicted labels for the testing feature set.
	 */
	public ArrayList<Integer> predict(ArrayList<int[]> trainingFeatures, ArrayList<Integer> trainingLabels,
			ArrayList<int[]> testingFeatures, int k) {
		ArrayList<Integer> predictedLabels = new ArrayList<>(); // Store predictions for testing features

		// Find the maximum label value to size the counting array
		int maxLabel = trainingLabels.stream().max(Integer::compare).orElse(0);

		// Iterate through each feature vector in the testing set
		for (int[] testingDataPoint : testingFeatures) {
			ArrayList<Neighbor> nearestNeighbors = new ArrayList<>();

			// Find the k-nearest neighbors
			for (int trainingDataPointIndex = 0; trainingDataPointIndex < trainingFeatures
					.size(); trainingDataPointIndex++) {
				// Calculate the Euclidean distance between the test point and the current
				// training point
				double distance = calculateEuclideanDistance(testingDataPoint,
						trainingFeatures.get(trainingDataPointIndex));

				// Store the neighbor and its distance
				nearestNeighbors.add(new Neighbor(trainingLabels.get(trainingDataPointIndex), distance));
			}

			// Sort neighbors by distance
			nearestNeighbors.sort((n1, n2) -> Double.compare(n1.distance, n2.distance));

			// Retrieve the k-nearest neighbors and count their labels
			int[] labelCounts = new int[maxLabel + 1]; // Array to count label occurrences
			for (int neighborIndex = 0; neighborIndex < k; neighborIndex++) {
				int label = nearestNeighbors.get(neighborIndex).label;
				labelCounts[label]++;
			}

			// Predict the label with the majority vote
			int predictedLabel = 0;
			int maxCount = 0;
			for (int label = 0; label < labelCounts.length; label++) {
				if (labelCounts[label] > maxCount) {
					maxCount = labelCounts[label];
					predictedLabel = label;
				}
			}

			predictedLabels.add(predictedLabel);
		}

		return predictedLabels; // Return the list of predictions
	}

	/**
	 * Calculates the Euclidean distance between two feature vectors. The Euclidean
	 * distance is defined as the square root of the sum of squared differences
	 * between corresponding elements of the two vectors.
	 *
	 * @param vector1 The first feature vector.
	 * @param vector2 The second feature vector.
	 * @return The Euclidean distance between the two feature vectors.
	 */
	private double calculateEuclideanDistance(int[] vector1, int[] vector2) {
		double squaredDifferenceSum = 0.0; // Initialise the sum of squared differences to zero

		// Iterate through each feature and compute the squared difference
		int len = vector1.length;
		for (int featureIndex = 0; featureIndex < vector1.length; featureIndex++) {
			squaredDifferenceSum += Math.pow(vector1[featureIndex] - vector2[featureIndex], 2);
		}

		// Return the square root of the sum to get the Euclidean distance
		return Math.sqrt(squaredDifferenceSum);
	}

	/**
	 * Helper class to store a neighbor's label and distance for sorting purposes.
	 */
	private static class Neighbor {
		int label;
		double distance;

		Neighbor(int label, double distance) {
			this.label = label;
			this.distance = distance;
		}
	}
}
