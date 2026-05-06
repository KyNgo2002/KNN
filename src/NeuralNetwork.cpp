#include "../include/NeuralNetwork.h"

#include <cmath>
#include <stdexcept>

// Constants
constexpr double cE = 2.718281828459045;

// Constructors
NeuralNetwork::NeuralNetwork(std::vector<size_t>& aLayers) {
    if (aLayers.size() < 2) {
        throw std::invalid_argument("NeuralNetwork must be initialized with at least 2 layers");
    }

    mLayers.reserve(aLayers.size());
    for (auto layerSize : aLayers) {
        if (layerSize < 1) {
            throw std::invalid_argument("Layer size must be at least 1");
        }
        mLayers.emplace_back(layerSize);
    }

    // Weights matrices are M x N
    //  - M: Size of next size of the next layer in the matrix
    //  - N: Size of the previous layer in the matrix
    mWeights.reserve(aLayers.size() - 1);
    for (size_t i = 1; i < aLayers.size(); ++i) {
       mWeights.emplace_back(Mat2D<double>(aLayers[i], aLayers[i - 1], 1.0));
    }
    
    mActivationFunctions.resize(aLayers.size(), ActivationFunction::Sigmoid);
}

size_t NeuralNetwork::size() const {
    return mLayers.size();
}

size_t NeuralNetwork::trainingDataSize() const {
    return mTrainingData.size();
}

size_t NeuralNetwork::trainingLabelsSize() const {
    return mTrainingLabels.size();
}

std::string NeuralNetwork::activationFunctionName(size_t ind) const { 
    if (ind >= mActivationFunctions.size()) {
        throw std::out_of_range("Index out of range for retrieving activation function name");
    }
    ActivationFunction activationFunction = mActivationFunctions[ind];
    switch (activationFunction) {
        case ActivationFunction::Sigmoid:
            return "Sigmoid";
		case ActivationFunction::ReLu:
			return "ReLu";
        default:
            throw std::invalid_argument("Unrecognized Activation Function");
    }
}

void NeuralNetwork::train() {
	train(mTrainingData.size());		
}

void NeuralNetwork::train(size_t numIterations) {
	if (mTrainingData.size() == 0 || mTrainingLabels.size() == 0) {
		throw std::logic_error("Training data or image labels must be provided before training");
	}

    if (mTrainingData.size() != mTrainingLabels.size()) {
        throw std::logic_error("Training data size does not match training labels size. Aborting");
    }
	std::cout << "Starting training" << std::endl;

	size_t numTrainingIterations = std::min(numIterations, mTrainingData.size());
	for (size_t imageIdx = 0; imageIdx < numTrainingIterations; ++imageIdx) {
		std::cout << "Iteration #:" << imageIdx + 1 << std::endl;
		// Forward pass
		std::cout << "Forward pass" << std::endl;
		forward(mTrainingData[imageIdx]);

		// Backpropagation
		std::cout << "Backpropagation" << std::endl;
	}	
}

void NeuralNetwork::setTrainingData(const std::vector<Mat2D<uint8_t>>& aTrainingData) {
    mTrainingData = aTrainingData;
}

void NeuralNetwork::setTrainingLabels(const std::vector<size_t>& aTrainingLabels) {
    mTrainingLabels = aTrainingLabels;
}

void NeuralNetwork::setActivationFunction(size_t ind, ActivationFunction aFunction) {
    if (ind >= mLayers.size()) {
        throw std::out_of_range("Invalid index provided to set activation function");
    }
    mActivationFunctions[ind] = aFunction;    
}

void NeuralNetwork::printLayer(size_t aLayerIdx) {
    if (aLayerIdx >= mLayers.size()) {
        throw std::out_of_range("Layer index out of bounds");
    }
}

void NeuralNetwork::forward(const Mat2D<uint8_t>& aInput) {
	if (aInput.mHeight != 1) {
		std::invalid_argument("Input to forward argument should be a 1 dimensional input matrix.");
	}
	std::cout << "Forward pass" << std::endl;

	vector<double> ioVector;
	// Propagate initial input through each weights layer in the neural network	
	for (size_t weightsMatIdx = 0; weightsMatIdx < mWeights.size(); ++weightsMatIdx) {
		layerOutput = mWeights[weightsMatIdx].multiply(aInput)	
	}
}

void NeuralNetwork::backpropagation() {
	std::cout << "Backward pass" << std::endl;
}

double NeuralNetwork::Sigmoid(double input) {
	return 1.0 / (1.0 + std::pow(cE, input));
}

double NeuralNetwork::ReLu(double input) {
	return std::max(0.0, input);
}

std::ostream& operator<<(std::ostream& aOut, NeuralNetwork& aNeuralNetwork) {
    aOut << "Neural Network parameters: " << std::endl;
    aOut << "Size: " << aNeuralNetwork.size() << std::endl;
    for (size_t i = 0; i < aNeuralNetwork.size(); ++i) {
        aOut << "---Layer " << i + 1 << "---" << std::endl;
        aOut << "Nodes: " << aNeuralNetwork.mLayers[i].size() << std::endl;
        aOut << "Weights: " << std::endl;
        if (i < aNeuralNetwork.size() - 1) {
            aOut << aNeuralNetwork.mWeights[i] << std::endl;
        }
        aOut << "Activation Function: " << aNeuralNetwork.activationFunctionName(i) << std::endl;
    }
    return aOut;
}

