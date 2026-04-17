#include <fstream>
#include <iostream>
#include <string>

#include "../include/Mat2D.h"
#include "../include/NeuralNetwork.h"
#include "../include/Reader.h"

int main() { 
    std::vector<size_t> layers = {2, 3, 3, 4, 5, 1};
    NeuralNetwork layeredNetwork(layers);

    const std::string filePath = "trunk/train-images-idx3-ubyte";

    readTrainingData(filePath); 
    
//    std::cout << layeredNetwork << std::endl;    
    return 0;    
}
