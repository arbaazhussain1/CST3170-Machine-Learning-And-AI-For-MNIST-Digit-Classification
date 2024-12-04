package machinelearningcoursework;

import java.util.ArrayList;
import java.util.List;

public class ClassifierEvaluator {

    /**
     * Evaluates the k-Nearest Neighbors (kNN) classifier.
     * 
     * @param knnClassifier      The kNN classifier instance.
     * @param trainingFeatureSet The training feature set.
     * @param trainingLabelSet   The training label set.
     * @param testingFeatureSet  The testing feature set.
     * @param testingLabelSet    The testing label set.
     * @return Accuracy of the kNN classifier as a percentage.
     */
    public double evaluateKNN(KNearestNeighbors knnClassifier, List<int[]> trainingFeatureSet, List<Integer> trainingLabelSet,
                              List<int[]> testingFeatureSet, List<Integer> testingLabelSet) {
        List<Integer> knnPredictedLabels = knnClassifier.predict(trainingFeatureSet, trainingLabelSet, testingFeatureSet);
        return calculateAccuracy(testingLabelSet, knnPredictedLabels);
    }

    /**
     * Evaluates a unified classifier using stacked predictions from kNN, SVM, and MLP.
     * 
     * @param trainingFeatureSet The training feature set.
     * @param trainingLabelSet   The training label set.
     * @param testingFeatureSet  The testing feature set.
     * @param testingLabelSet    The testing label set.
     * @param svmClassifier      The Support Vector Machine (SVM) classifier.
     * @param mlpClassifier      The Multi-Layer Perceptron (MLP) classifier.
     * @return Accuracy of the unified classifier as a percentage.
     */
    public double evaluateUnifiedClassifier(List<int[]> trainingFeatureSet, List<Integer> trainingLabelSet,
                                            List<int[]> testingFeatureSet, List<Integer> testingLabelSet,
                                            SupportVectorMachineClassifier svmClassifier, MultiLayerPerceptronClassifier mlpClassifier) {
        // Get predictions from kNN
        List<Integer> knnPredictions = new KNearestNeighbors().predict(trainingFeatureSet, trainingLabelSet, testingFeatureSet);
        
        // Get predictions from SVM and MLP
        List<Integer> svmPredictions = new ArrayList<>();
        List<Integer> mlpPredictions = new ArrayList<>();

        for (int[] testSample : testingFeatureSet) {
            svmPredictions.add(svmClassifier.predict(testSample));
            mlpPredictions.add(mlpClassifier.predict(testSample));
        }

        // Create meta-features for stacking
        List<int[]> stackedMetaFeatures = new ArrayList<>();
        for (int testingSampleIndex = 0; testingSampleIndex < testingFeatureSet.size(); testingSampleIndex++) {
            stackedMetaFeatures.add(new int[]{knnPredictions.get(testingSampleIndex), 
                                               svmPredictions.get(testingSampleIndex), 
                                               mlpPredictions.get(testingSampleIndex)});
        }

        // Train a meta-classifier using the stacked meta-features
        SupportVectorMachineClassifier metaClassifier = new SupportVectorMachineClassifier(3, 0.01, 1.0, 1000);
        metaClassifier.train(stackedMetaFeatures, testingLabelSet);

        // Get predictions from the meta-classifier
        List<Integer> unifiedClassifierPredictions = new ArrayList<>();
        for (int[] metaFeatureSample : stackedMetaFeatures) {
            unifiedClassifierPredictions.add(metaClassifier.predict(metaFeatureSample));
        }

        // Calculate accuracy of the unified classifier
        return calculateAccuracy(testingLabelSet, unifiedClassifierPredictions);
    }

    /**
     * Calculates the accuracy of predictions against true labels.
     * 
     * @param actualLabels   The true labels.
     * @param predictedLabels The predicted labels.
     * @return Accuracy as a percentage.
     */
    public double calculateAccuracy(List<Integer> actualLabels, List<Integer> predictedLabels) {
        int correctPredictionCount = 0;
        for (int labelIndex = 0; labelIndex < actualLabels.size(); labelIndex++) {
            if (actualLabels.get(labelIndex).equals(predictedLabels.get(labelIndex))) {
                correctPredictionCount++;
            }
        }
        return (correctPredictionCount * 100.0) / actualLabels.size();
    }
}
