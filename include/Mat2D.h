#ifndef MAT2D_H
#define MAT2D_H

#include <concepts>
#include <iomanip>
#include <iostream>
#include <type_traits>
#include <vector>

template <typename T>
concept Numeric = std::is_arithmetic_v<T>;

template <Numeric T>
class Mat2D {
// Matrix class
public:
    Mat2D() = default;
    Mat2D(size_t aWidth, size_t aHeight);
    Mat2D(size_t aWidth, size_t aHeight, T aValue);
    Mat2D(const Mat2D& other);

    Mat2D<T> add(const Mat2D<T>& aOther);
    std::vector<T> multiply(const std::vector<T>& aVec);

    template <Numeric U>
    friend std::ostream& operator<<(std::ostream& aOut, Mat2D<U>& aMatrix);

    std::vector<T>& operator[](size_t row);
private:
    size_t mWidth;
    size_t mHeight;
    std::vector<std::vector<T>> mData;
};

template <Numeric T>
Mat2D<T>::Mat2D(size_t aWidth, size_t aHeight) {
    mWidth = aWidth;
    mHeight = aHeight;
    mData.resize(aHeight, std::vector<T>(aWidth));
}

template <Numeric T>
Mat2D<T>::Mat2D(size_t aWidth, size_t aHeight, T value) {
    mWidth = aWidth;
    mHeight = aHeight;
    mData.resize(aHeight, std::vector<T>(aWidth, value));
}

template <Numeric T>
Mat2D<T>::Mat2D(const Mat2D& other) {
    mWidth = other.mWidth;
    mHeight = other.mHeight;
    mData.resize(mHeight, std::vector<T>(mWidth));
    for (size_t i = 0; i < mHeight; ++i) {
        for (size_t j = 0; j < mWidth; ++j) {
            mData[i][j] = other.mData[i][j];
        }
    }
}

template <Numeric T>
Mat2D<T> Mat2D<T>::add(const Mat2D<T>& aOther) {
    if (aOther.width() != mWidth || aOther.height() != mHeight) {
       throw std::invalid_argument("Attempting to add Mat2D with matrix that does not match the current matrix"); 
    }
}

template <Numeric T>
std::vector<T> Mat2D<T>::multiply(const std::vector<T>& aVec) {
    if (aVec.size() != mWidth) {
        throw std::length_error("Attempting to multiply Mat2D with incorrectly sized vector.");
    }

    std::vector<T> output(mHeight);
    for (size_t row = 0; row < mHeight; ++row) {
        for (size_t col = 0; col < mWidth; ++col) {
           output[row] += aVec[row] * mData[row][col]; 
        }
    }
    return output;
}

template <Numeric U>
std::ostream& operator<<(std::ostream& aOut, Mat2D<U>& aMatrix) {
    for (const auto& row : aMatrix.mData) {
        std::cout << "[ "; 
        for (auto value : row) {
            std::cout << std::setw(3) << static_cast<int>(value) << " ";
        }        
        std::cout << "]" << std::endl;
    }
    return aOut;
}

template <Numeric T>
std::vector<T>& Mat2D<T>::operator[](size_t row) {
    if (row >= mHeight) {
        throw std::out_of_range("Invalid query into matrix");  
    }
    return mData[row];
}
#endif
