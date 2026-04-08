#include <fstream>
#include <iostream>
#include <string>

#include "../include/Mat2D.h"
#include "../include/NeuralNetwork.h"

int main() {

    std::string file = "trunk/train-labels-idx1-ubyte"; 
    std::ifstream fstream(file, std::ios::binary);

    if (fstream.is_open() == false) {
        std::cout << "File failed to open" << std::endl;
    }
    
    std::vector<size_t> layers = {2, 3, 3, 4, 5, 1};
    NeuralNetwork layeredNetwork(layers);
    
//    std::cout << layeredNetwork << std::endl;    
    return 0;    
}
