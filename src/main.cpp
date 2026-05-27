#include <fstream>
#include <iostream>
#include <string>

#include "../include/Mat2D.h"
#include "../include/NeuralNetwork.h"
#include "../include/Reader.h"

int main() { 
    const std::string trainingDataFilePath = "trunk/train-images-idx3-ubyte";
    const std::string trainingLabelsFilePath = "trunk/train-labels-idx1-ubyte";
    const std::string testingImagesFilePath = "trunk/t10k-iamges-idx3-ubyte";
    const std::string testingLabelsFilepath = "trunk/t10k-labels-idx1-ubyte";

	// Reading training data 
    std::vector<Mat2D<uint8_t>> trainingImages = readImageData(trainingDataFilePath); 
    std::vector<size_t> trainingLabels = readImageLabels(trainingLabelsFilePath);
    
    const uint32_t imageVectorSize = trainingImages[0].width() * trainingImages[0].height();

    std::vector<size_t> layers = {imageVectorSize, 392, 196, 98, 44, 22, 10};
    NeuralNetwork layeredNetwork(layers);

    layeredNetwork.setTrainingData(trainingImages);
    layeredNetwork.setTrainingLabels(trainingLabels);

    std::cout << "Training Data: " << std::endl; 
    std::cout << "Number of Images: " << layeredNetwork.trainingDataSize() << std::endl;    
    std::cout << "Number of Labels: " << layeredNetwork.trainingLabelsSize() << std::endl;

    std::cout << layeredNetwork << std::endl;

	// Training	
    size_t iterations = 1;
	layeredNetwork.train(iterations);

    // Read testing data
    //std::vector<Mat2D<uint8_t>> testingImages = readImageData(testingImagesFilePath);
    //std::vector<size_t> testingLabels = readImageLabels(trainingLabelsFilePath);

    // Testing
    //layeredNetwork.test(testingImages, testingLabels);
    return 0;    
}
