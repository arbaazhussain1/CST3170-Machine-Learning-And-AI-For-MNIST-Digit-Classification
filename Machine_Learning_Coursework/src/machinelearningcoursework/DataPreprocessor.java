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
     * @param labelList     List to store labels.
     */
    public void loadFileData(String filePath, List<int[]> featureMatrix, List<Integer> labelList) {
        try (Scanner fileScanner = new Scanner(new File(filePath))) {
            while (fileScanner.hasNextLine()) {
                String line = fileScanner.nextLine();
                String[] elements = line.split(",");

                if (elements.length == 65) { // 64 features + 1 label
                    int[] featureRow = new int[64];
                    for (int featureIndex = 0; featureIndex < 64; featureIndex++) {
                        featureRow[featureIndex] = Integer.parseInt(elements[featureIndex]);
                    }
                    featureMatrix.add(featureRow); // Add the features to the feature matrix
                    labelList.add(Integer.parseInt(elements[64])); // Add the label to the label list
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
        for (int[] features : featureMatrix) {
            // Calculate the L2 norm of the feature vector
            double vectorNorm = Math.sqrt(Arrays.stream(features).mapToDouble(value -> value * value).sum());
            if (vectorNorm > 0) { // Avoid division by zero
                for (int featureIndex = 0; featureIndex < features.length; featureIndex++) {
                    features[featureIndex] = (int) Math.round((features[featureIndex] / vectorNorm) * 1000);
                }
            }
        }
    }

    /**
     * Splits a dataset into two folds for cross-validation.
     * 
     * @param features        Original feature dataset.
     * @param labels          Original label dataset.
     * @param fold1Features   List to store features for the first fold.
     * @param fold1Labels     List to store labels for the first fold.
     * @param fold2Features   List to store features for the second fold.
     * @param fold2Labels     List to store labels for the second fold.
     */
    public void splitDataset(List<int[]> features, List<Integer> labels,
                             List<int[]> fold1Features, List<Integer> fold1Labels,
                             List<int[]> fold2Features, List<Integer> fold2Labels) {
        int midpoint = features.size() / 2; // Split the dataset in half

        for (int dataIndex = 0; dataIndex < features.size(); dataIndex++) {
            if (dataIndex < midpoint) {
                fold1Features.add(features.get(dataIndex));
                fold1Labels.add(labels.get(dataIndex));
            } else {
                fold2Features.add(features.get(dataIndex));
                fold2Labels.add(labels.get(dataIndex));
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
        for (int labelIndex = 0; labelIndex < labels.size(); labelIndex++) {
            labels.set(labelIndex, labels.get(labelIndex) == targetClass ? 1 : -1);
        }
    }
}
