#ifndef READER_H
#define READER_H

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "Mat2D.h"

uint32_t readFourBytes(std::ifstream& fstream, uint8_t* byte) {
    fstream.read(reinterpret_cast<char*>(byte), 4);
    return (byte[0] << 24) | (byte[1] << 16) | (byte[2] << 8) | (byte[3]);
}

std::vector<int> readTrainingData(const std::string& filePath) {
    std::cout << "Attempting to read file: " << filePath << std::endl;
    std::ifstream fstream(filePath, std::ios::binary);

    if (fstream.is_open() == false) {
        std::cout << "File failed to open" << std::endl;
    }

    uint8_t bytes[4];
    uint32_t something = readFourBytes(fstream, bytes);
    size_t numDimensions = static_cast<int>(bytes[3]);
    std::cout << "Dimensions: " << numDimensions << std::endl;
    std::vector<size_t> dimensions(numDimensions);
     
    for (size_t i = 0; i < numDimensions; ++i) {
        dimensions[i] = readFourBytes(fstream, bytes); 
        std::cout << "Dimension " << i + 1 << ": " << dimensions[i] << std::endl;
    }
    
    uint8_t byte = 0; 
    // std::vector<Mat2D<uint8_t>> images(dimensions[0], Mat2D<uint8_t>(dimensions[1], dimensions[2]));
    Mat2D<uint8_t> test(28, 28, 0);
    for (size_t i = 0; i < dimensions[1]; ++i) {
        for (size_t j = 0; j < dimensions[2]; ++j) {
//            fstream.read(reinterpret_cast<char*>(test[i][j]), 1);
            fstream.read(reinterpret_cast<char*>(&byte), 1); 
            test[i][j] = byte;
        }   
    }        

    std::cout << test << std::endl;

    return {};
}

#endif
