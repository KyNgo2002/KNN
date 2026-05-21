#ifndef UTIL_H
#define UTIL_H

#include <vector>

namespace Util {

template <typename T, typename U>
std::vector<T> subtract(const std::vector<T>& aFirst, const std::vector<U>& aSecond) {
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
std::vector<T> dot(const std::vector<T>& aFirst, const std::vector<T>& aSecond) {
    if (aFirst.size() != aSecond.size()) {
        std::string errorMessage = "Vector Dot Product: Size of both input vectors must be equal.\nFirst vector size: " + std::to_string(aFirst.size()) + "\nSecond vector size: " + std::to_string(aSecond.size()) + "\n";  
        throw std::invalid_argument(errorMessage);
    }
    std::vector<T> output(aFirst.size());
    for (size_t i = 0; i < aFirst.size(); ++i) {
        output[i] = aFirst[i] * aSecond[i];
    }
    return output;
}

template <typename T>
std::vector<T> scalar(std::vector<double>& aVec, double scalar) {
    std::vector<T> output(aVec.size());
    for (size_t i = 0; i < aVec.size(); ++i) {
        output[i] = aVec[i] * scalar;
    } 
}

} // namespace Util

#endif
