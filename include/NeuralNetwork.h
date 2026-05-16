#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include "Layer.h"
#include "Mat2D.h"

#include <cstdint>
#include <iostream>
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
    
    void setTrainingData(const std::vector<Mat2D<uint8_t>>& aTrainingData);
    void setTrainingLabels(const std::vector<size_t>& aTrainingLabels);

    void setActivationFunction(size_t ind, ActivationFunction aFunction);

    void printLayer(size_t aLayerIdx);

    friend std::ostream& operator<<(std::ostream& aOut, NeuralNetwork& aNeuralNetwork);

private:
    std::vector<double> forward(const std::vector<uint8_t>& aInput);
    void backpropagation(const std::vector<double>& aForwardOutput);

    static ActFunc getActivationFunction(ActivationFunction aActivationFunction);

	static double Sigmoid(double aInput, bool aDerivative = false);
	static double ReLu(double aInput, bool aDerivative = false);
   
    static std::vector<double> Softmax(const std::vector<double>& aInput, bool aDerivative = false);
 
    std::vector<Mat2D<uint8_t>> mTrainingData;
    std::vector<size_t> mTrainingLabels;

    std::vector<Layer<double>> mLayers;
    std::vector<Mat2D<double>> mWeights;
    std::vector<ActivationFunction> mActivationFunctions;
};

#endif
