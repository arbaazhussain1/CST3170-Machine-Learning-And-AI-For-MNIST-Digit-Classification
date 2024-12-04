package machinelearningcoursework;

import java.util.ArrayList;
import java.util.List;

public class MachineLearningPipeline {

    public static void main(String[] args) {
        // Paths to dataset files
        String dataset1Path = "dataSet1.csv"; // Path to the first dataset
        String dataset2Path = "dataSet2.csv"; // Path to the second dataset

        // Data structures to hold features and labels for both datasets
        List<int[]> trainingFeaturesDataset1 = new ArrayList<>();
        List<int[]> trainingFeaturesDataset2 = new ArrayList<>();
        List<Integer> trainingLabelsDataset1 = new ArrayList<>();
        List<Integer> trainingLabelsDataset2 = new ArrayList<>();

        // Instantiate the DataPreprocessor for handling preprocessing tasks
        DataPreprocessor preprocessor = new DataPreprocessor();

        // Load the datasets into feature and label lists
        preprocessor.loadFileData(dataset1Path, trainingFeaturesDataset1, trainingLabelsDataset1);
        preprocessor.loadFileData(dataset2Path, trainingFeaturesDataset2, trainingLabelsDataset2);

        // Normalise the features to ensure consistent scaling
        preprocessor.normaliseFeatures(trainingFeaturesDataset1);
        preprocessor.normaliseFeatures(trainingFeaturesDataset2);

        // Prepare data structures for cross-validation folds
        List<int[]> fold1Features = new ArrayList<>();
        List<int[]> fold2Features = new ArrayList<>();
        List<Integer> fold1Labels = new ArrayList<>();
        List<Integer> fold2Labels = new ArrayList<>();

        // Split dataset1 into two folds for cross-validation
        preprocessor.splitDataset(trainingFeaturesDataset1, trainingLabelsDataset1, fold1Features, fold1Labels, fold2Features, fold2Labels);

        // Convert labels for SVM (binary classification: target class = 1, others = -1)
        preprocessor.convertLabelsForSVM(fold1Labels, 1);
        preprocessor.convertLabelsForSVM(fold2Labels, 1);

        // Instantiate the ClassifierEvaluator to handle evaluation tasks
        ClassifierEvaluator evaluator = new ClassifierEvaluator();
        System.out.println("\n--- Evaluation Results ---");

        // k-Nearest Neighbors (kNN) evaluation
        KNearestNeighbors knn = new KNearestNeighbors();
        double knnAccuracy = evaluator.evaluateKNN(knn, fold1Features, fold1Labels, fold2Features, fold2Labels);
        System.out.printf("kNN Accuracy: %.2f%%\n", knnAccuracy);

        // Support Vector Machine (SVM) evaluation with grid search for hyper-parameter tuning
        SupportVectorMachineClassifier svm = new SupportVectorMachineClassifier(64, 0.005, 1.0, 10000);
        svm.gridSearch(fold1Features, fold1Labels, fold2Features, fold2Labels);
        double svmAccuracy = svm.evaluate(fold2Features, fold2Labels);
        System.out.printf("SVM Accuracy: %.2f%%\n", svmAccuracy);

        // Multi-Layer Perceptron (MLP) evaluation
        MultiLayerPerceptronClassifier mlp = new MultiLayerPerceptronClassifier(64, 128, 1, 0.0001, 2000);
        mlp.train(fold1Features, fold1Labels); // Train the MLP on fold1
        double mlpAccuracy = mlp.evaluate(fold2Features, fold2Labels); // Evaluate the MLP on fold2
        System.out.printf("MLP Accuracy: %.2f%%\n", mlpAccuracy);

        // Unified Classifier evaluation (stacking ensemble method)
        double unifiedAccuracy = evaluator.evaluateUnifiedClassifier(fold1Features, fold1Labels, fold2Features, fold2Labels, svm, mlp);
        System.out.printf("Unified Classifier Accuracy: %.2f%%\n", unifiedAccuracy);
    }
}
