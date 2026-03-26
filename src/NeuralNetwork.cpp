#include "../include/NeuralNetwork.h"

#include <stdexcept>

// Constructors
NeuralNetwork::NeuralNetwork(std::vector<size_t>& aLayers) {
    mLayers.reserve(aLayers.size());
    for (auto layerSize : aLayers) {
        mLayers.emplace_back(std::max(layerSize, static_cast<size_t>(1)));
    }
}

size_t NeuralNetwork::size() const {
    return mLayers.size();
}

void NeuralNetwork::printLayer(size_t aLayerIdx) {
    if (aLayerIdx >= mLayers.size()) {
        throw std::out_of_range("Layer index out of bounds");
    }
}

std::ostream& operator<<(std::ostream& aOut, NeuralNetwork& aNeuralNetwork) {
    aOut << "Neural Network parameters: " << std::endl;
    aOut << "Size: " << aNeuralNetwork.size() << std::endl;
    size_t layerIndex = 1;
    for (const auto& layer : aNeuralNetwork.mLayers) {
        aOut << "---Layer " << layerIndex++ << "---" << std::endl;
        aOut << "Nodes: " << layer.size() << std::endl;
    }
    return aOut;
}

