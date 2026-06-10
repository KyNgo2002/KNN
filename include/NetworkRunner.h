#include <memory>

#include "Mat2D.h"
#include "NeuralNetwork.h"

class NetworkRunner {
public:
    NetworkRunner(NeuralNetwork* aNetwork);

    void run();
    
private:
    void printMenu();

    // Members
    NeuralNetwork* mNetwork;
};
