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
    
	// Reading training data 
    std::vector<Mat2D<uint8_t>> images = readImageData(trainingDataFilePath); 
    std::vector<size_t> labels = readImageLabels(trainingLabelsFilePath);

    layeredNetwork.setTrainingData(images);
    layeredNetwork.setTrainingLabels(labels);

	layeredNetwork.setActivationFunction(0, NeuralNetwork::ActivationFunction::ReLu);

    std::cout << "Training Data: " << std::endl; 
    std::cout << "Number of Images: " << layeredNetwork.trainingDataSize() << std::endl;    
    std::cout << "Number of Labels: " << layeredNetwork.trainingLabelsSize() << std::endl;

    std::cout << layeredNetwork << std::endl;

	// Training	
	layeredNetwork.train(1000);
    return 0;    
}
