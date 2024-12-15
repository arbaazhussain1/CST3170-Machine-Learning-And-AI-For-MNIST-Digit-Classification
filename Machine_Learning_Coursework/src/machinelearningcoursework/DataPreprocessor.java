package machinelearningcoursework;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

/**
 * Handles preprocessing of input datasets. Includes loading data from CSV
 * files, normalising features, splitting datasets, and converting labels for
 * binary classification.
 */
public class DataPreprocessor {

	// Constants for dataset dimensions
	private static final int TOTAL_DIMENSIONS = 65; // Total dimensions including features and label
	private static final int FEATURE_DIMENSIONS = 64; // Number of feature dimensions
	private static final String CSV_DELIMITER = ","; // Delimiter used in the CSV file

	
	// Reads feature and label data from a CSV file into provided arrays, ensuring
	// proper resource closure with try-with-resources.
	public void loadFileData(String filePath, int[][] featureMatrix, int[] labelsList, boolean debug) {
		// Using try-with-resources to ensure BufferedReader is properly closed after
		// use
		try (BufferedReader reader = new BufferedReader(new FileReader(filePath))) {
			String line; // Temporary variable to store each line of the CSV
			int rowIndex = 0; // Counter to track the current row being read
			// Read the file line by line
			while ((line = reader.readLine()) != null) {
				// Split the line into individual elements based on the CSV delimiter
				String[] elements = line.split(CSV_DELIMITER); // Use the constant for splitting

				// Ensure the line has the expected number of elements before processing
				if (elements.length == TOTAL_DIMENSIONS) {
					try {
						// Parse feature values and store them in the featureMatrix
						for (int featureIndex = 0; featureIndex < FEATURE_DIMENSIONS; featureIndex++) {
							featureMatrix[rowIndex][featureIndex] = Integer.parseInt(elements[featureIndex]);
						}
						// Parse the label and store it in the labelsList
						labelsList[rowIndex] = Integer.parseInt(elements[FEATURE_DIMENSIONS]);
						rowIndex++; // Move to the next row
					} catch (NumberFormatException error) { // Log an error if a number format issue is encountered in
															// debug mode
						if (debug) {
							System.err.println("Invalid number format in line: " + line);
						}
					}
				} else if (debug) {
					// Log an error if the line does not match the expected format in debug mode
					System.err.println("Line skipped due to unexpected format: " + line);
				}
			}
			System.out.println("Total rows loaded: " + rowIndex); // Log the total number of rows successfully loaded

		} catch (IOException error) { // Log an error if an issue occurs while reading the file
			System.err.println("File error: " + error.getMessage());
		}
	}
}
