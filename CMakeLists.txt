cmake_minimum_required(VERSION 3.28)

# Project and executable name
project(KNN)

# Set CMake variables
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(KNN
    src/main.cpp
    src/NetworkRunner.cpp
    src/NeuralNetwork.cpp
    src/Mat2D.cpp
)

target_include_directories(KNN PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)
