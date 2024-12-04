package machinelearningcoursework;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;

public class DataPreprocessor {

    /**
     * Loads feature and label data from a CSV file.
     * 
     * @param filePath     Path to the CSV file.
     * @param featureMatrix List to store feature rows.
     * @param labelsList     List to store labels.
     */
    public void loadFileData(String filePath, List<int[]> featureMatrix, List<Integer> labelsList) {
        try (Scanner fileScanner = new Scanner(new File(filePath))) {
            while (fileScanner.hasNextLine()) {
                String line = fileScanner.nextLine();
                String[] elements = line.split(",");

                if (elements.length == 65) { // 64 features + 1 label
                    int[] featureVector = new int[64];
                    for (int featureIndex = 0; featureIndex < 64; featureIndex++) {
                        featureVector[featureIndex] = Integer.parseInt(elements[featureIndex]);
                    }
                    featureMatrix.add(featureVector); // Add the features to the feature matrix
                    labelsList.add(Integer.parseInt(elements[64])); // Add the label to the label list
                } else {
                    System.out.println("Line skipped due to unexpected format.");
                }
            }
        } catch (FileNotFoundException e) {
            System.err.println("File not found: " + e.getMessage());
        }
    }

    /**
     * Normalises feature vectors in a matrix using L2 normalisation.
     * 
     * @param featureMatrix List of feature vectors to normalise.
     */
    public void normaliseFeatures(List<int[]> featureMatrix) {
        for (int[] featureVector : featureMatrix) {
            // Calculate the L2 norm of the feature vector
            double l2Norm = Math.sqrt(Arrays.stream(featureVector).mapToDouble(value -> value * value).sum());
            if (l2Norm > 0) { // Avoid division by zero
                for (int featureIndex = 0; featureIndex < featureVector.length; featureIndex++) {
                    featureVector[featureIndex] = (int) Math.round((featureVector[featureIndex] / l2Norm) * 1000);
                }
            }
        }
    }

    /**
     * Splits a dataset into two folds for cross-validation.
     * 
     * @param features        Original feature dataset.
     * @param labels          Original label dataset.
     * @param firstFoldFeatures   List to store features for the first fold.
     * @param firstFoldLabels     List to store labels for the first fold.
     * @param secondFoldFeatures   List to store features for the second fold.
     * @param secondFoldLabels     List to store labels for the second fold.
     */
    public void splitDataset(List<int[]> features, List<Integer> labels,
                             List<int[]> firstFoldFeatures, List<Integer> firstFoldLabels,
                             List<int[]> secondFoldFeatures, List<Integer> secondFoldLabels) {
        int midpoint = features.size() / 2; // Split the dataset in half

        for (int dataPointIndex = 0; dataPointIndex < features.size(); dataPointIndex++) {
            if (dataPointIndex < midpoint) {
                firstFoldFeatures.add(features.get(dataPointIndex));
                firstFoldLabels.add(labels.get(dataPointIndex));
            } else {
                secondFoldFeatures.add(features.get(dataPointIndex));
                secondFoldLabels.add(labels.get(dataPointIndex));
            }
        }
    }

    /**
     * Converts labels for binary classification by setting the target class to 1
     * and all other classes to -1.
     * 
     * @param labels      List of labels to convert.
     * @param targetClass The class to set as positive (1).
     */
    public void convertLabelsForSVM(List<Integer> labels, int targetClass) {
        for (int labelPosition = 0; labelPosition < labels.size(); labelPosition++) {
            labels.set(labelPosition, labels.get(labelPosition) == targetClass ? 1 : -1);
        }
    }
}
