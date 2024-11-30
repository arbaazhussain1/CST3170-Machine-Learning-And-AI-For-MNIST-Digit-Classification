package machinelearningcoursework;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class MachineLearningCoursework {

    public static void main(String[] args) {

        // Paths to datasets
        String file1Path = "dataSet1.csv";
        String file2Path = "dataSet2.csv";

        // Load datasets
        List<int[]> data1 = new ArrayList<>();
        List<int[]> data2 = new ArrayList<>();
        List<Integer> categories1 = new ArrayList<>();
        List<Integer> categories2 = new ArrayList<>();

        loadFileData(file1Path, data1, categories1);
        loadFileData(file2Path, data2, categories2);

        // Perform two-fold cross-validation
        System.out.println("\nTwo-fold cross-validation:");

        // Fold 1: Train on data1, Test on data2
        double accuracyFold1 = evaluate(data1, categories1, data2, categories2);

        // Fold 2: Train on data2, Test on data1
        double accuracyFold2 = evaluate(data2, categories2, data1, categories1);

        // Average accuracy
        double averageAccuracy = (accuracyFold1 + accuracyFold2) / 2;

        // Print results
        System.out.printf("The Fold 1 Accuracy: %.2f%%\n", accuracyFold1);
        System.out.printf("The Fold 2 Accuracy: %.2f%%\n", accuracyFold2);
        System.out.printf("The Average Accuracy: %.2f%%\n", averageAccuracy);
    }

    private static void loadFileData(String filePath, List<int[]> data, List<Integer> categories) {
        try (Scanner scanner = new Scanner(new File(filePath))) {
            while (scanner.hasNextLine()) {
                String line = scanner.nextLine();
                String[] elements = line.split(",");

                if (elements.length == 65) {
                    int[] digitData = new int[64];
                    for (int i = 0; i < 64; i++) {
                        digitData[i] = Integer.parseInt(elements[i]);
                    }
                    data.add(digitData);
                    categories.add(Integer.parseInt(elements[64]));
                } else {
                    System.out.println("Line skipped due to unexpected format.");
                }
            }
        } catch (FileNotFoundException e) {
            System.err.println("File not found: " + e.getMessage());
        }
    }

    private static double evaluate(List<int[]> trainData, List<Integer> trainCategories, List<int[]> testData, List<Integer> testCategories) {
        // Categorize the test dataset based on the training dataset
        List<Integer> predictedCategories = categorizeData(trainData, trainCategories, testData);

        // Calculate accuracy
        return calculateAccuracy(testCategories, predictedCategories);
    }

    private static List<Integer> categorizeData(List<int[]> data1, List<Integer> categories1, List<int[]> data2) {
        List<Integer> assignedCategories = new ArrayList<>();

        for (int[] row2 : data2) {
            int closestIndex = -1;
            double minDistance = Double.MAX_VALUE;

            for (int i = 0; i < data1.size(); i++) {
                double distance = calculateEuclideanDistance(row2, data1.get(i));

                if (distance < minDistance) {
                    minDistance = distance;
                    closestIndex = i;
                }
            }

            assignedCategories.add(categories1.get(closestIndex));
        }

        return assignedCategories;
    }

    private static double calculateAccuracy(List<Integer> trueCategories, List<Integer> assignedCategories) {
        if (trueCategories.size() != assignedCategories.size()) {
            throw new IllegalArgumentException("Mismatch in size of true and assigned categories.");
        }

        int correctCount = 0;
        for (int i = 0; i < trueCategories.size(); i++) {
            if (trueCategories.get(i).equals(assignedCategories.get(i))) {
                correctCount++;
            }
        }

        return (correctCount * 100.0) / trueCategories.size();
    }

    private static double calculateEuclideanDistance(int[] row1, int[] row2) {
        double sum = 0.0;

        for (int i = 0; i < 64; i++) { // Only comparing the first 64 elements
            sum += Math.pow(row1[i] - row2[i], 2);
        }

        return Math.sqrt(sum);
    }
}
