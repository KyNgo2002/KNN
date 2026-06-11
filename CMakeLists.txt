cmake_minimum_version(VERSION 3.28)

# Project and executable name
project(KNN)
add_executable(KNN)

# Set CMake variables
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(KNN
    src/main.cpp
    src/NetworkRunner.cpp
    src/NeuralNetwork.cpp
    src/Mat2D.cpp
)

target_include_directories(KNN private
    ${PROJECT_SOURCE_DIR}/include
)
