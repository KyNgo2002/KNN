#ifndef NEURALNETWORK_H
#define NERUALNETWORK_H

#include "Layer.h"
#include "Mat2D.h"

#include <iostream>
#include <vector>

class NeuralNetwork {
public:
    NeuralNetwork() = default;
    NeuralNetwork(std::vector<size_t>& aLayers);

    size_t size() const;

    void train();

    void printLayer(size_t aLayerIdx);

    friend std::ostream& operator<<(std::ostream& aOut, NeuralNetwork& aNeuralNetwork);

private:
    void forward();
    void backpropagation();

    std::vector<Layer<double>> mLayers;
    std::vector<Mat2D<double>> mWeights;
};

#endif
