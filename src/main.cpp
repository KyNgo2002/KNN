#include <fstream>
#include <iostream>
#include <string>

#include "../include/Mat2D.h"
#include "../include/NeuralNetwork.h"
#include "../include/Reader.h"

int main() { 
    std::vector<size_t> layers = {2, 3, 3, 4, 5, 1};
    NeuralNetwork layeredNetwork(layers);

    const std::string trainingDataFilePath = "trunk/train-images-idx3-ubyte";
    const std::string trainingLabelsFilePath = "trunk/train-labels-idx1-ubyte";
    
    std::vector<Mat2D<uint8_t>> images = readFile(trainingDataFilePath); 
    std::vector<Mat2D<uint8_t>> labels = readFile(trainingLabelsFilePath);

    layeredNetwork.setTrainingData(images);
    layeredNetwork.setTrainingLabels(labels);

    std::cout << "Training Data: " << std::endl; 
    std::cout << "Number of Images: " << layeredNetwork.trainingDataSize() << std::endl;    
    std::cout << "Number of Labels: " << layeredNetwork.trainingLabelsSize() << std::endl;
    return 0;    
}
