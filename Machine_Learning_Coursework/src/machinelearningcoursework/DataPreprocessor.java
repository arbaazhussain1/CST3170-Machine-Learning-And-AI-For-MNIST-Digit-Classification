package machinelearningcoursework;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Scanner;

/**
 * Handles preprocessing of input datasets. Includes loading data from CSV
 * files, normalising features, splitting datasets, and converting labels for
 * binary classification.
 */
public class DataPreprocessor {

	// Constants for dataset dimensions
	private static final int TOTAL_DIMENSIONS = 65; // Total dimensions including features and label
	private static final int FEATURE_DIMENSIONS = 64; // Number of feature dimensions

	/**
	 * Loads feature and label data from a CSV file.
	 *
	 * @param filePath      Path to the CSV file.
	 * @param featureMatrix ArrayList to store feature rows (features).
	 * @param labelsList    ArrayList to store labels.
	 */
	/**
	 * Loads feature and label data from a CSV file.
	 *
	 * @param filePath      Path to the CSV file.
	 * @param featureMatrix ArrayList to store feature rows (features).
	 * @param labelsList    ArrayList to store labels.
	 * @param debug         Flag to enable detailed error messages for debugging.
	 */
	public void loadFileData(String filePath, ArrayList<int[]> featureMatrix, ArrayList<Integer> labelsList,
			boolean debug) {
		try (Scanner fileScanner = new Scanner(new File(filePath))) {
			while (fileScanner.hasNextLine()) {
				String line = fileScanner.nextLine(); // Read the next line
				String[] elements = line.split(","); // Split the line by commas into an array of strings

				if (elements.length == TOTAL_DIMENSIONS) { // Expecting 64 features + 1 label
					try {
						int[] featureVector = new int[FEATURE_DIMENSIONS]; // Array to hold the 64 features
						for (int featureIndex = 0; featureIndex < FEATURE_DIMENSIONS; featureIndex++) {
							featureVector[featureIndex] = Integer.parseInt(elements[featureIndex]); // Parse each
																									// feature as an
																									// integer
						}
						featureMatrix.add(featureVector); // Add the feature vector to the feature matrix
						labelsList.add(Integer.parseInt(elements[FEATURE_DIMENSIONS])); // Parse and add the label
					} catch (NumberFormatException error) {
						if (debug) {
							// Log an error if number parsing fails
							System.err.println("Invalid number format in line: " + line);
						}
					}
				} else {
					if (debug) {
						// Log an error if the line format is unexpected
						System.err.println("Line skipped due to unexpected format: " + line);
					}
				}
			}
		} catch (FileNotFoundException error) {
			// Handle the case where the file does not exist
			System.err.println("File not found: " + filePath);
		}
	}

	/**
	 * Normalises feature vectors using L2 normalisation. Scales each feature vector
	 * so that its Euclidean norm (L2 norm) is 1.
	 *
	 * @param featureMatrix ArrayList of feature vectors to normalise.
	 */
	public void normaliseFeatures(ArrayList<int[]> featureMatrix) {
		for (int[] features : featureMatrix) {
			// Calculate the L2 norm of the feature vector
			double l2Norm = Math.sqrt(Arrays.stream(features).mapToDouble(value -> value * value).sum());
			if (l2Norm > 0) { // Avoid division by zero
				for (int featureIndex = 0; featureIndex < features.length; featureIndex++) {
					// Scale each feature and multiply by 1000 to retain precision
					features[featureIndex] = (int) Math.round((features[featureIndex] / l2Norm) * 1000);
				}
			}
		}
	}

	/**
	 * Splits a dataset into two equal-sized folds for cross-validation.
	 * 
	 * The first half of the dataset is assigned to Fold 1, and the second half is
	 * assigned to Fold 2.
	 *
	 * @param features      Original feature dataset as an ArrayList.
	 * @param labels        Original label dataset as an ArrayList.
	 * @param fold1Features ArrayList for storing features of the first fold.
	 * @param fold1Labels   ArrayList for storing labels of the first fold.
	 * @param fold2Features ArrayList for storing features of the second fold.
	 * @param fold2Labels   ArrayList for storing labels of the second fold.
	 */
	public void splitDataset(ArrayList<int[]> features, ArrayList<Integer> labels, ArrayList<int[]> fold1Features,
			ArrayList<Integer> fold1Labels, ArrayList<int[]> fold2Features, ArrayList<Integer> fold2Labels) {
		int midpoint = features.size() / 2; // Calculate the midpoint for splitting

		for (int dataPointIndex = 0; dataPointIndex < features.size(); dataPointIndex++) {
			if (dataPointIndex < midpoint) {
				// Assign the first half of the data to Fold 1
				fold1Features.add(features.get(dataPointIndex));
				fold1Labels.add(labels.get(dataPointIndex));
			} else {
				// Assign the second half of the data to Fold 2
				fold2Features.add(features.get(dataPointIndex));
				fold2Labels.add(labels.get(dataPointIndex));
			}
		}
	}

	/**
	 * Converts labels for binary classification. Assigns a positive class (1) to
	 * the target class and negative class (-1) to all other classes.
	 *
	 * @param labels      ArrayList of labels to convert.
	 * @param targetClass The class to be marked as positive (1).
	 */
//	public void convertLabelsForSVM(ArrayList<Integer> labels, int targetClass) {
//		for (int labelPosition = 0; labelPosition < labels.size(); labelPosition++) {
//			// Update the label: 1 if it matches the target class, -1 otherwise
//			labels.set(labelPosition, labels.get(labelPosition) == targetClass ? 1 : -1);
//		}
//	}
}
