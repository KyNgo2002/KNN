#include <iostream>

#include "../include/NetworkRunner.h"

NetworkRunner::NetworkRunner(NeuralNetwork* aNetwork) {
    if (!aNetwork) {
        throw std::invalid_argument("Network Runner construction failed: Provided network was null\n");
    }
    mNetwork = aNetwork;
}

void NetworkRunner::run() {
    std::cout << "\033[2J";
    std::cout << "\nWelcome to the Network Runner: \n\n";
    char userInput;
    while (userInput != '3') {
        printMenu();
        std::cin >> userInput;
        switch (userInput) {
            case '1':
                std::cout << *mNetwork << std::endl;
                break;
            case '2': {
                std::string input = "";
                while (input != "Q") {
                    std::cout << "Please enter an image index between 0 and " << mNetwork->testingSetSize() << " Or enter 'Q' to quit: ";
                    std::cin >> input;
                    std::cout << "\033[2J";
                    if (input == "Q" || input == "q") {
                        break;
                    }
                    size_t value = 0;
                    try {
                        value = stoull(input);
                        mNetwork->printSingleTestImage(value);
                    }
                    catch(std::exception&) {
                        std::cout << "Invalid Input. Try again" << std::endl;
                        continue;
                    }
                    size_t output = mNetwork->testSingleImage(value);
                    std::cout << "Network output: " << output << std::endl;
                }
                break;
            }
            case '3':
                std::cout << "Thanks for using! Quitting!" << std::endl;
                break;
            default:
                std::cout << "Invalid user input. Try again" << std::endl;
                break;
        }
    }
}

void NetworkRunner::printMenu() {
    std::cout << "1). Print Neural Network parameters" << std::endl;
    std::cout << "2). Test individual images" << std::endl;
    std::cout << "3). Quit" << std::endl;
    std::cout << "Please select one of the available options: ";
}
