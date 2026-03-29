#ifndef MAT2D_H
#define MAT2D_H

#include <iostream>
#include <vector>

template <typename T>
class Mat2D {
// Matrix class
public:
    Mat2D() = default;
    Mat2D(size_t aWidth, size_t aHeight);
    Mat2D(size_t aWidth, size_t aHeight, T aValue);

    std::vector<T> multiply(const std::vector<T>& aVec);

    template <typename U>
    friend std::ostream& operator<<(std::ostream& aOut, Mat2D<U>& aMatrix);

    std::vector<T>& operator[](size_t row);
private:
    size_t mWidth;
    size_t mHeight;
    std::vector<std::vector<T>> mData;
};

template <typename T>
Mat2D<T>::Mat2D(size_t aWidth, size_t aHeight) {
    mWidth = aWidth;
    mHeight = aHeight;
    mData.resize(aHeight, std::vector<T>(aWidth));
}

template <typename T>
Mat2D<T>::Mat2D(size_t aWidth, size_t aHeight, T value) {
    mWidth = aWidth;
    mHeight = aHeight;
    mData.resize(aHeight, std::vector<T>(aWidth, value));
}

template <typename T>
std::vector<T> Mat2D<T>::multiply(const std::vector<T>& aVec) {
    
}

template <typename U>
std::ostream& operator<<(std::ostream& aOut, Mat2D<U>& aMatrix) {
    for (const auto& row : aMatrix.mData) {
        std::cout << "[ "; 
        for (auto value : row) {
            std::cout << value << " ";
        }        
        std::cout << "]" << std::endl;
    }
    return aOut;
}

template <typename T>
std::vector<T>& Mat2D<T>::operator[](size_t row) {
    if (row >= mHeight) {
        throw std::out_of_range("Invalid query into matrix");  
    }
    return mData[row];
}
#endif
