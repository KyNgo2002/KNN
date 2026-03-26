#ifndef NEURALNETWORK_H
#define NERUALNETWORK_H

#include "Layer.h"

#include <iostream>
#include <vector>

class NeuralNetwork {
public:
    NeuralNetwork() = default;
    NeuralNetwork(std::vector<size_t>& aLayers);

    size_t size() const;

    void printLayer(size_t aLayerIdx);

    friend std::ostream& operator<<(std::ostream& aOut, NeuralNetwork& aNeuralNetwork);

private:
    std::vector<Layer> mLayers;
};

#endif
