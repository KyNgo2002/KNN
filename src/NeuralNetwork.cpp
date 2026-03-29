#include "../include/NeuralNetwork.h"

#include <stdexcept>

// Constructors
NeuralNetwork::NeuralNetwork(std::vector<size_t>& aLayers) {
    if (aLayers.size() < 2) {
        throw std::invalid_argument("NeuralNetwork must be initialized with at least 2 layers");
    }

    mLayers.reserve(aLayers.size());
    for (auto layerSize : aLayers) {
        mLayers.emplace_back(std::max(layerSize, static_cast<size_t>(1)));
    }
        
    mWeights.reserve(aLayers.size() - 1);
    for (size_t i = 1; i < aLayers.size(); ++i) {
       mWeights.emplace_back(Mat2D<double>(aLayers[i], aLayers[i - 1], 1.0));
    }
}

size_t NeuralNetwork::size() const {
    return mLayers.size();
}

void NeuralNetwork::forward() {

}

void NeuralNetwork::printLayer(size_t aLayerIdx) {
    if (aLayerIdx >= mLayers.size()) {
        throw std::out_of_range("Layer index out of bounds");
    }
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

