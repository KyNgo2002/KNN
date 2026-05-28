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

    NeuralNetwork() = default;
    NeuralNetwork(std::vector<size_t>& aLayers);

    size_t size() const;
    size_t trainingDataSize() const;
    size_t trainingLabelsSize() const;
    std::string activationFunctionName(size_t ind) const;

	void train();
    void train(size_t numIterations);

    void test(const std::vector<Mat2D<double>>& aImages, const std::vector<size_t>& aLabels);

    void setEpochs(size_t aEpochs);

    void setLearningRate(double aLearningRate);
 
    void setTrainingData(const std::vector<Mat2D<double>>& aTrainingData);
    void setTrainingLabels(const std::vector<size_t>& aTrainingLabels);

    void setActivationFunction(size_t ind, ActivationFunction aFunction);

    friend std::ostream& operator<<(std::ostream& aOut, NeuralNetwork& aNeuralNetwork);

private:
    std::vector<double> forward(const std::vector<double>& aInput);
    void backpropagation(size_t aCorrectDigit);

    static ActFunc getActivationFunction(ActivationFunction aActivationFunction);
    
    static std::vector<double> costFunction(const std::vector<double>& aInput, size_t aCorrectDigit);

	static double Sigmoid(double aInput, bool aDerivative = false);
	static double ReLu(double aInput, bool aDerivative = false);
   
    static std::vector<double> Softmax(const std::vector<double>& aInput);

    void randomizeMatrix(Mat2D<double>& aMat);

    std::mt19937 mGenerator;
    std::uniform_real_distribution<double> mDist;

    size_t mEpochs;
    double mLearningRate; 
 
    std::vector<Mat2D<double>> mTrainingData;
    std::vector<size_t> mTrainingLabels;

    // Layer outputs without activation function applied.
    std::vector<std::vector<double>> mLayerOutputs;
    // Layer outputs with activation function applied.
    std::vector<std::vector<double>> mLayerOutputsTransformed;
    std::vector<Mat2D<double>> mWeights;
    std::vector<ActivationFunction> mActivationFunctions;
};

#endif
