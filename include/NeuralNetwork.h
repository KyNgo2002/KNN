#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include "Mat2D.h"

#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

class NeuralNetwork {
public:
    using ActFunc = double(*)(double, bool);

    enum class ActivationFunction {
        Sigmoid,        
        ReLu,
        None,
    };

    NeuralNetwork(const std::vector<size_t>& aLayers);
    NeuralNetwork(const std::string& aModelFilePath);
    NeuralNetwork(size_t aEpochs, double aLearningRate, const std::vector<Mat2D<double>>& aWeights, const std::vector<std::vector<double>>& aBiases, const std::vector<ActivationFunction>& aActivationFunctions);

    size_t size() const;
    size_t trainingSetSize() const;
    size_t testingSetSize() const;
    std::string getActivationFunctionName(size_t ind) const;

	void train();
    void train(size_t numIterations);

    void test();
    void test(size_t aIterations);
    size_t testSingleImage(size_t aImageIdx);

    void printSingleTestImage(size_t aImageIdx);

    static NeuralNetwork readModel(const std::string& aModelFilePath);
    void writeModel(const std::string& aModelFilePath);

    void setEpochs(size_t aEpochs);

    void setLearningRate(double aLearningRate);
 
    void setTrainingData(const std::vector<Mat2D<double>>& aTrainingData);
    void setTrainingLabels(const std::vector<size_t>& aTrainingLabels);

    void setTestingData(const std::vector<Mat2D<double>>& aTestingData);
    void setTestingLabels(const std::vector<size_t>& aTestingLabels);

    void setActivationFunction(size_t ind, ActivationFunction aFunction);

    static void Softmax(std::vector<double>& aInput);

    friend std::ostream& operator<<(std::ostream& aOut, const NeuralNetwork& aNeuralNetwork);

private:
    std::vector<double> forward(const std::vector<double>& aInput);
    void backpropagation(size_t aCorrectDigit);

    static ActFunc getActivationFunction(ActivationFunction aActivationFunction);
    
    static double costFunction(const std::vector<double>& aInput, size_t aCorrectDigit);

	static double Sigmoid(double aInput, bool aDerivative = false);
    static double ReLu(double aInput, bool aDerivative);
   
    void randomizeMatrix(Mat2D<double>& aMat);

    std::mt19937 mGenerator;
    std::normal_distribution<double> mDist;

    size_t mEpochs;
    double mLearningRate; 
 
    std::vector<Mat2D<double>> mTrainingData;
    std::vector<size_t> mTrainingLabels;

    std::vector<Mat2D<double>> mTestingData;
    std::vector<size_t> mTestingLabels;

    // Layer outputs without activation function applied.
    std::vector<std::vector<double>> mLayerOutputs;
    // Layer outputs with activation function applied.
    std::vector<std::vector<double>> mLayerOutputsTransformed;
    std::vector<Mat2D<double>> mWeights;
    std::vector<std::vector<double>> mBiases;
    std::vector<ActivationFunction> mActivationFunctions;
};

#endif
