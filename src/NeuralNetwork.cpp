#include "../include/NeuralNetwork.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

// Constructors
NeuralNetwork::NeuralNetwork(std::vector<size_t>& aLayers) {
    if (aLayers.size() < 2) {
        throw std::invalid_argument("NeuralNetwork must be initialized with at least 2 layers");
    }

    mLayers.reserve(aLayers.size());
    for (auto layerSize : aLayers) {
        if (layerSize < 1) {
            throw std::invalid_argument("Layer size must be at least 1");
        }
        mLayers.emplace_back(layerSize);
    }

    // Weights matrices are M x N
    //  - M: Size of next size of the next layer in the matrix
    //  - N: Size of the previous layer in the matrix
    mWeights.reserve(aLayers.size() - 1);
    for (size_t i = 1; i < aLayers.size(); ++i) {
       mWeights.emplace_back(Mat2D<double>(aLayers[i - 1], aLayers[i], 1.0));
    }
    
    mActivationFunctions.resize(aLayers.size(), ActivationFunction::Sigmoid);

    // First and last layers do not have activation functions.
    mActivationFunctions[0] = mActivationFunctions[aLayers.size() - 1] = ActivationFunction::None;   
}

size_t NeuralNetwork::size() const {
    return mLayers.size();
}

size_t NeuralNetwork::trainingDataSize() const {
    return mTrainingData.size();
}

size_t NeuralNetwork::trainingLabelsSize() const {
    return mTrainingLabels.size();
}

std::string NeuralNetwork::activationFunctionName(size_t ind) const { 
    if (ind >= mActivationFunctions.size()) {
        throw std::out_of_range("Index out of range for retrieving activation function name");
    }
    ActivationFunction activationFunction = mActivationFunctions[ind];
    switch (activationFunction) {
        case ActivationFunction::None:
            return "None";
        case ActivationFunction::Sigmoid:
            return "Sigmoid";
		case ActivationFunction::ReLu:
			return "ReLu";
        default:
            throw std::invalid_argument("Unrecognized Activation Function");
    }
}

void NeuralNetwork::train() {
	train(mTrainingData.size());		
}

void NeuralNetwork::train(size_t numIterations) {
	if (mTrainingData.size() == 0 || mTrainingLabels.size() == 0) {
		throw std::logic_error("Training data or image labels must be provided before training");
	}

    if (mTrainingData.size() != mTrainingLabels.size()) {
        throw std::logic_error("Training data size does not match training labels size. Aborting");
    }
	std::cout << "Starting training:" << std::endl;

	size_t numTrainingIterations = std::min(numIterations, mTrainingData.size());
	for (size_t imageIdx = 0; imageIdx < numTrainingIterations; ++imageIdx) {
		std::cout << "\n------Training Iteration #" << imageIdx + 1 << "------" << std::endl;
		// Forward pass
		std::vector<double> forwardPassResult = forward(mTrainingData[imageIdx].toVec());

		// Backpropagation
         
	}	
}
	
void NeuralNetwork::setTrainingData(const std::vector<Mat2D<uint8_t>>& aTrainingData) {
    mTrainingData = aTrainingData;
}

void NeuralNetwork::setTrainingLabels(const std::vector<size_t>& aTrainingLabels) {
    mTrainingLabels = aTrainingLabels;
}

void NeuralNetwork::setActivationFunction(size_t ind, ActivationFunction aFunction) {
    if (ind >= mLayers.size()) {
        throw std::out_of_range("Invalid index provided to set activation function");
    }
    if (ind == 0 || ind == mActivationFunctions.size() - 1) {
        std::cout << "Activation function may not be set for the first and last layers in the network" << std::endl;
        return;
    }
    mActivationFunctions[ind] = aFunction;    
}

void NeuralNetwork::printLayer(size_t aLayerIdx) {
    if (aLayerIdx >= mLayers.size()) {
        throw std::out_of_range("Layer index out of bounds");
    }
}

std::vector<double> NeuralNetwork::forward(const std::vector<uint8_t>& aInput) {
	std::cout << "\nForward pass\n" << std::endl;

	std::vector<double> ioVector(aInput.begin(), aInput.end());
	// Propagate initial input through each weights layer in the neural network	
	for (size_t weightsMatIdx = 0; weightsMatIdx < mWeights.size(); ++weightsMatIdx) {
        std::cout << "---Forward Progress " << weightsMatIdx + 1 << "/" << mWeights.size() << "---" << std::endl;
        std::cout << "Input Vector size: " << ioVector.size() << std::endl;
        std::cout << "Weights Matrix size: " << mWeights[weightsMatIdx].size() << std::endl; 
        
        // Multiply weights matrix with input vector
		ioVector = mWeights[weightsMatIdx].multiply(ioVector);	

        // Retrieve Activation Function for current layer
        const auto activationFunction = getActivationFunction(mActivationFunctions[weightsMatIdx]);
        // Apply activation function to each output in the current layer.
        for (size_t ind = 0; ind < ioVector.size(); ++ind) {
            ioVector[ind] = activationFunction ? activationFunction(ioVector[ind], false) : ioVector[ind];
        }
	}
    return ioVector;
}

void NeuralNetwork::backpropagation(const std::vector<double>& aForwardOutput) {
	std::cout << "Backward pass" << std::endl;
}

NeuralNetwork::ActFunc NeuralNetwork::getActivationFunction(ActivationFunction aActivationFunction) {
    switch (aActivationFunction) {
        case ActivationFunction::None:
            return nullptr;
        case ActivationFunction::Sigmoid:
            return Sigmoid;
        case ActivationFunction::ReLu:
            return ReLu;
        default:
            std::cerr << "Chosen activation function does not have a corresponding implementation" << std::endl;
    }
    return nullptr;
}

double NeuralNetwork::Sigmoid(double aInput, bool aDerivative) {
    if (aDerivative) {
        return std::exp(-aInput) / std::pow(1 + std::exp(-aInput), 2);
    }
	return 1.0 / (1.0 + std::exp(aInput));
}

double NeuralNetwork::ReLu(double aInput, bool aDerivative) {
    if (aDerivative) {
        return aInput;
    }
	return std::max(0.0, aInput);
}

std::vector<double> NeuralNetwork::Softmax(const std::vector<double>& aInput, bool aDerivative) {
    if (aDerivative) {
        return aInput;
    }
    std::vector<double> output(aInput.size());

    double maxElement = *max_element(aInput.begin(), aInput.end());
    double sum = 0;
    
    // Calculate exponentiated of input vector.
    for (size_t i = 0; i < aInput.size(); ++i) {
        output[i] = std::exp(aInput[i] - maxElement);
        sum += output[i]; 
    }

    // Divide individual exponentiated values by exponentiated sum to bind
    // sum of output to 1.
    std::for_each(output.begin(), output.end(), [sum](double& element) {
        element /= sum;
    });

    return output;
}

std::ostream& operator<<(std::ostream& aOut, NeuralNetwork& aNeuralNetwork) {
    aOut << "Neural Network parameters: " << std::endl;
    aOut << "Size: " << aNeuralNetwork.size() << std::endl;
    for (size_t i = 0; i < aNeuralNetwork.size(); ++i) {
        aOut << "---Layer " << i + 1 << "---" << std::endl;
        aOut << "Nodes: " << aNeuralNetwork.mLayers[i].size() << std::endl;
        aOut << "Weights: " << std::endl;
        if (i < aNeuralNetwork.size() - 1) {
            aOut << aNeuralNetwork.mWeights[i] << std::endl;
        }
        aOut << "Activation Function: " << aNeuralNetwork.activationFunctionName(i) << std::endl << std::endl;
    }
    return aOut;
}

