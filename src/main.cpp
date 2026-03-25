#include <iostream>
#include "../include/Node.h"

int main() {
    std::cout << "Hello world" << std::endl;

    Node* firstNode = new Node();
    Node* secondNode = new Node();

    firstNode->setNext(secondNode);
    return 0;
}
