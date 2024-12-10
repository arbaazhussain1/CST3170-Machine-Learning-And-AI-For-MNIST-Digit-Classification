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
	 * @return An ArrayList of predicted labels for the testing feature set.
	 */
	public ArrayList<Integer> predict(ArrayList<int[]> trainingFeatures, ArrayList<Integer> trainingLabels,
			ArrayList<int[]> testingFeatures) {
		ArrayList<Integer> predictedLabels = new ArrayList<>(); // Store predictions for testing features

		// Iterate through each feature vector in the testing set
		for (int[] testingDataPoint : testingFeatures) {
			double nearestDistance = Double.MAX_VALUE; // Initialise the nearest distance to a large value
			int labelOfNearestNeighbor = -1; // Placeholder for the label of the nearest neighbor

			// Find the nearest neighbor from the training set
			for (int trainingDataPointIndex = 0; trainingDataPointIndex < trainingFeatures
					.size(); trainingDataPointIndex++) {
				// Calculate the Euclidean distance between the test point and the current
				// training point
				double distance = calculateEuclideanDistance(testingDataPoint,
						trainingFeatures.get(trainingDataPointIndex));

				// Update nearest neighbor if the current distance is smaller
				if (distance < nearestDistance) {
					nearestDistance = distance; // Update the nearest distance
					labelOfNearestNeighbor = trainingLabels.get(trainingDataPointIndex); // Store the corresponding
																							// label
				}
			}

			// Add the predicted label of the nearest neighbor to the results
			predictedLabels.add(labelOfNearestNeighbor);
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
		for (int featureIndex = 0; featureIndex < vector1.length; featureIndex++) {
			squaredDifferenceSum += Math.pow(vector1[featureIndex] - vector2[featureIndex], 2);
		}

		// Return the square root of the sum to get the Euclidean distance
		return Math.sqrt(squaredDifferenceSum);
	}
}
