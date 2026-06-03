#ifndef UTIL_H
#define UTIL_H

#include <vector>

namespace Util {

template <typename T, typename U>
std::vector<T> Subtract(const std::vector<T>& aFirst, const std::vector<U>& aSecond) {
    if (aFirst.size() != aSecond.size()) {
        throw std::invalid_argument("Vector Subtraction: Size of both input vectors must be equal");
    }
    std::vector<T> output(aFirst.size());
    for (size_t i = 0; i < aFirst.size(); ++i) {
        output[i] = aFirst[i] - aSecond[i];
    }   
    return output;
}

template <typename T>
std::vector<T> multiply(const std::vector<T>& aFirst, const std::vector<T>& aSecond) {
    if (aFirst.size() != aSecond.size()) {
        std::string errorMessage = "Vector Multiply: Size of both input vectors must be equal.\nFirst vector size: " + std::to_string(aFirst.size()) + "\nSecond vector size: " + std::to_string(aSecond.size()) + "\n";  
        throw std::invalid_argument(errorMessage);
    }
    std::vector<T> output(aFirst.size());
    for (size_t i = 0; i < aFirst.size(); ++i) {
        output[i] = aFirst[i] * aSecond[i];
    }
    return output;
}

template <typename T>
Mat2D<T> VecToMatrix(const std::vector<T>& aFirst, const std::vector<T>& aSecond) {
    Mat2D<T> output(aFirst.size(), aSecond.size());
    for (size_t i = 0; i < aFirst.size(); ++i) {
        for (size_t j = 0; j < aSecond.size(); ++j) {
            output[i][j] = aFirst[i] * aSecond[j];
        }
    }
    return output;
}

template <typename T>
std::vector<T> Scalar(std::vector<T>& aVec, double scalar) {
    std::vector<T> output(aVec.size());
    for (size_t i = 0; i < aVec.size(); ++i) {
        output[i] = aVec[i] * scalar;
    } 
}

template <typename T>
void Print(const std::vector<T>& aVec) {
    std::cout << "[ ";
    for (size_t ind = 0; ind < aVec.size(); ++ind) {
        std::cout << std::setprecision(10) << static_cast<double>(aVec[ind]) << " ";
    }
    std::cout << "]" << std::endl;
}

} // namespace Util

#endif
