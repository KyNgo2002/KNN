#ifndef NEURALNETWORK_H
#define NERUALNETWORK_H

#include "Layer.h"
#include "Mat2D.h"

#include <cstdint>
#include <iostream>
#include <vector>

class NeuralNetwork {
public:
    NeuralNetwork() = default;
    NeuralNetwork(std::vector<size_t>& aLayers);

    size_t size() const;
    size_t trainingDataSize() const;
    size_t trainingLabelsSize() const;

    void train();
    
    void setTrainingData(const std::vector<Mat2D<uint8_t>>& aTrainingData);
    void setTrainingLabels(const std::vector<Mat2D<uint8_t>>& aTrainingLabels);

    void printLayer(size_t aLayerIdx);

    friend std::ostream& operator<<(std::ostream& aOut, NeuralNetwork& aNeuralNetwork);

private:
    void forward();
    void backpropagation();

    std::vector<Mat2D<uint8_t>> mTrainingData;
    std::vector<Mat2D<uint8_t>> mTrainingLabels;

    std::vector<Layer<double>> mLayers;
    std::vector<Mat2D<double>> mWeights;
};

#endif
