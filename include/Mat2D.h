#ifndef MAT2D_H
#define MAT2D_H

#include <concepts>
#include <iomanip>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

template <typename T>
concept Numeric = std::is_arithmetic_v<T>;

template <Numeric T>
class Mat2D {
// Matrix class
public:
    Mat2D() = default;
    Mat2D(size_t aHeight, size_t aWidth);
    Mat2D(size_t aHeight, size_t aWidth, T aValue);
    Mat2D(const Mat2D<T>& aOther);
    Mat2D<T>& operator=(const Mat2D<T>& aOther);

    size_t width() const;
    size_t height() const;
    std::string size() const;    

    Mat2D<T> add(const Mat2D<T>& aOther);
    std::vector<T> multiply(const std::vector<T>& aVec);
    Mat2D<T> scalar(double aScalar);
	
    Mat2D<T> transpose();
	std::vector<T> toVec() const;

    std::vector<T>& operator[](size_t row);
    const std::vector<T>& operator[](size_t row) const;

    Mat2D<T> operator-(const Mat2D<T>& aMat);

    template <Numeric U>
    friend std::ostream& operator<<(std::ostream& aOut, const Mat2D<U>& aMatrix);

private:
    size_t mHeight;
    size_t mWidth;
    std::vector<std::vector<T>> mData;
};

template <Numeric T>
Mat2D<T>::Mat2D(size_t aHeight, size_t aWidth) {
    mHeight = aHeight;
    mWidth = aWidth;
    mData.resize(aHeight, std::vector<T>(aWidth));
}

template <Numeric T>
Mat2D<T>::Mat2D(size_t aHeight, size_t aWidth, T value) {
    mHeight = aHeight;
    mWidth = aWidth;
    mData.resize(aHeight, std::vector<T>(aWidth, value));
}

template <Numeric T>
Mat2D<T>& Mat2D<T>::operator=(const Mat2D<T>& aOther) {
    mHeight = aOther.mHeight;
    mWidth = aOther.mWidth;
    mData.assign(mHeight, std::vector<T>(mWidth));
    for (size_t i = 0; i < mHeight; ++i) {
        for (size_t j = 0; j < mWidth; ++j) {
            mData[i][j] = aOther[i][j];
        }
    }
    return *this;
}

template <Numeric T>
Mat2D<T>::Mat2D(const Mat2D<T>& aOther) {
    mWidth = aOther.mWidth;
    mHeight = aOther.mHeight;
    mData.resize(mHeight, std::vector<T>(mWidth));
    for (size_t i = 0; i < mHeight; ++i) {
        for (size_t j = 0; j < mWidth; ++j) {
            mData[i][j] = aOther.mData[i][j];
        }
    }
}

template <Numeric T>
size_t Mat2D<T>::width() const {
    return mWidth;
}

template <Numeric T>
size_t Mat2D<T>::height() const {
    return mHeight;
}

template <Numeric T>
std::string Mat2D<T>::size() const {
    return std::to_string(mHeight) + "x" + std::to_string(mWidth);
}

template <Numeric T>
Mat2D<T> Mat2D<T>::add(const Mat2D<T>& aOther) {
    if (aOther.mWidth != mWidth || aOther.mHeight != mHeight) {
       throw std::invalid_argument("Attempting to add Mat2D with matrix that does not match the current matrix"); 
    }
}

template <Numeric T>
std::vector<T> Mat2D<T>::multiply(const std::vector<T>& aVec) {
    if (aVec.size() != mWidth) {
        std::string errorMessage = "Attempting to multiply Mat2D with incorrectly sized vector.\nInput Vector size: " + std::to_string(aVec.size()) + "\nInput Matrix size: " + this->size() + "\n";
        throw std::length_error(errorMessage);
    }

    std::vector<T> output(mHeight);
    for (size_t row = 0; row < mHeight; ++row) {
        for (size_t col = 0; col < mWidth; ++col) {
            output[row] += aVec[col] * mData[row][col]; 
        }
    }
    return output;
}

template <Numeric T>
Mat2D<T> Mat2D<T>::scalar(double scalar) {
    for (auto& row : mData) {
        for (T& elem : row) {
            elem = elem * scalar;
        }
    }
    return *this; 
}

template <Numeric T>
Mat2D<T> Mat2D<T>::transpose() {
    Mat2D<T> output(mWidth, mHeight);
    for (size_t i = 0; i < mHeight; ++i) {
        for (size_t j = 0; j < mWidth; ++j) {
            output[j][i] = mData[i][j];  
        }
    }
    return output;
}

template <Numeric T>
std::vector<T> Mat2D<T>::toVec() const {
	std::vector<T> output(mWidth * mHeight);
	for (size_t row = 0; row < mHeight; ++row) {
		for (size_t col = 0; col < mWidth; ++col) {
			output[row * mWidth + col] = mData[row][col];
		}
	} 
	return output;
}

template <Numeric T>
std::vector<T>& Mat2D<T>::operator[](size_t row) {
    if (row >= mHeight) {
        throw std::out_of_range("Invalid query into matrix");  
    }
    return mData[row];
}

template <Numeric T>
const std::vector<T>& Mat2D<T>::operator[](size_t row) const {
    if (row >= mHeight) {
        throw std::out_of_range("Invalid query into matrix");  
    }
    return mData[row];
}


template <Numeric T>
Mat2D<T> Mat2D<T>::operator-(const Mat2D<T>& aMat) {
    if (mHeight != aMat.height() || mWidth != aMat.width()) {
        std::string errorMessage = "Invalid matrix subtraction. Matrix sizes must match to perform subtraction.\n";
        errorMessage += "First matrix size: " + size() + "\n";
        errorMessage += "Second matrix size: " + aMat.size() + "\n";
        throw std::invalid_argument(errorMessage);
    }
    Mat2D<T> output(mHeight, mWidth);
    for (size_t i = 0; i < mHeight; ++i) {
        for (size_t j = 0; j < mWidth; ++j) {
            output[i][j] = mData[i][j] - aMat[i][j]; 
        }
    }
    return output;
}

template <Numeric U>
std::ostream& operator<<(std::ostream& aOut, const Mat2D<U>& aMatrix) {
    for (const auto& row : aMatrix.mData) {
        std::cout << "[ "; 
        for (auto value : row) {
            std::cout << std::setw(5) << std::setprecision(3) << static_cast<double>(value) << " ";
        }        
        std::cout << "]" << std::endl;
    }
    return aOut;
}

#endif
