#include "../include/NeuralNetwork.h"

#include <stdexcept>

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

void NeuralNetwork::train() {
    if (mTrainingData.size() != mTrainingLabels.size()) {
        throw std::logic_error("Training data size does not match training labels size. Aborting");
    }
}

void NeuralNetwork::setTrainingData(const std::vector<Mat2D<uint8_t>>& aTrainingData) {
    mTrainingData = aTrainingData;
}

void NeuralNetwork::setTrainingLabels(const std::vector<Mat2D<uint8_t>>& aTrainingLabels) {
    mTrainingLabels = aTrainingLabels;
}

void NeuralNetwork::printLayer(size_t aLayerIdx) {
    if (aLayerIdx >= mLayers.size()) {
        throw std::out_of_range("Layer index out of bounds");
    }
}

void NeuralNetwork::forward() {

}

void NeuralNetwork::backpropagation() {

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
    }
    return aOut;
}

