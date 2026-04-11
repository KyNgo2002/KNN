#ifndef READER_H
#define READER_H

#include <iostream>
#include <string>
#include <vector>

std::vector<int> readTrainingData(const std::string& filePath) {
    std::ifstream fstream(filePath, std::ios::binary);

    if (fstream.is_open() == false) {
        std::cout << "File failed to open" << std::endl;
    }

    int test = 0;
    
    while (fstream >> test) {
        std::cout << test << std::endl;
    }

    return {};
}

#endif
