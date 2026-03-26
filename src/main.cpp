#include <iostream>
#include "../include/NeuralNetwork.h"

int main() {
    std::vector<size_t> layers = {2, 3, 3, 4, 0, 1};
    NeuralNetwork layeredNetwork(layers);
    
    std::cout << layeredNetwork << std::endl;    
    return 0;
}
