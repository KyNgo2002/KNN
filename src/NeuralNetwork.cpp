#include "../include/NeuralNetwork.h"
#include "../include/Util.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

// Constructors
NeuralNetwork::NeuralNetwork(std::vector<size_t>& aLayers) {
    if (aLayers.size() < 2) {
        throw std::invalid_argument("NeuralNetwork must be initialized with at least 2 layers");
    }

    mEpochs = 1;
    mLearningRate = 0.1;
    
    mLayerOutputs.reserve(aLayers.size());
    mLayerOutputs.reserve(aLayers.size());
    for (size_t layerSize : aLayers) {
        if (layerSize < 1) {
            throw std::invalid_argument("Layer size must be at least 1");
        }
        mLayerOutputs.emplace_back(layerSize); 
        mLayerOutputsTransformed.emplace_back(layerSize); 
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
    return mLayerOutputs.size();
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
        std::vector<double> output = forward(mTrainingData[imageIdx].toVec());

		// Backpropagation
        backpropagation(mTrainingLabels[imageIdx]); 
	}	
}

void NeuralNetwork::setEpochs(size_t aEpochs) {
    mEpochs = aEpochs;
}

void NeuralNetwork::setLearningRate(double aLearningRate) {
    if (aLearningRate <= 0) {
        std::cout << "Learning rate must be larger than 0.0" << std::endl;
    }
    mLearningRate = aLearningRate;
}

void NeuralNetwork::setTrainingData(const std::vector<Mat2D<uint8_t>>& aTrainingData) {
    mTrainingData = aTrainingData;
}

void NeuralNetwork::setTrainingLabels(const std::vector<size_t>& aTrainingLabels) {
    mTrainingLabels = aTrainingLabels;
}

void NeuralNetwork::setActivationFunction(size_t ind, ActivationFunction aFunction) {
    if (ind >= mLayerOutputs.size()) {
        throw std::out_of_range("Invalid index provided to set activation function");
    }
    if (ind == 0 || ind == mActivationFunctions.size() - 1) {
        std::cout << "Activation function may not be set for the first and last layers in the network" << std::endl;
        return;
    }
    mActivationFunctions[ind] = aFunction;    
}

std::vector<double> NeuralNetwork::forward(const std::vector<uint8_t>& aInput) {
	std::cout << "\nForward pass\n" << std::endl;

    mLayerOutputs[0] = std::vector<double>(aInput.begin(), aInput.end());
    mLayerOutputsTransformed[0] = std::vector<double>(aInput.begin(), aInput.end());
    // Propagate initial input through each weights layer in the neural network	
	for (size_t weightsMatIdx = 0; weightsMatIdx < mWeights.size(); ++weightsMatIdx) {
        std::cout << "---Forward Progress " << weightsMatIdx + 1 << "/" << mWeights.size() << "---" << std::endl;
        std::cout << "Input Vector size: " << aInput.size() << std::endl;
        std::cout << "Weights Matrix size: " << mWeights[weightsMatIdx].size() << std::endl; 
        
        std::vector<double>& currOutput = mLayerOutputsTransformed[weightsMatIdx + 1]; 

        // Multiply weights matrix with input vector
		currOutput = mWeights[weightsMatIdx].multiply(mLayerOutputsTransformed[weightsMatIdx]);

        // Save current output as untransformed layer output
        mLayerOutputs[weightsMatIdx] = currOutput;

        // Retrieve Activation Function for current layer
        const auto activationFunction = getActivationFunction(mActivationFunctions[weightsMatIdx]);
    
        // Apply activation function to each output in the current layer.
        for (double& element : currOutput) {
            if (activationFunction) {
                element = activationFunction(element, false);
            }
        } 
	}
    return Softmax(mLayerOutputsTransformed.back()); 
}

void NeuralNetwork::backpropagation(size_t aCorrectDigit) {
	std::cout << "\nBackward pass" << std::endl;
   
    std::vector<double> delta; 
    std::vector<uint8_t> expected(10, 0);
    expected[aCorrectDigit] = 1;
    for (int idx = mLayerOutputs.size() - 1; idx >= 1; --idx) {
        std::cout << "---Backward Progress " << mLayerOutputs.size() - idx << "/" << mLayerOutputs.size() - 1 << "---" << std::endl;
        delta = Util::subtract<double, uint8_t>(mLayerOutputs[idx], expected);
    } 
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

std::vector<double> NeuralNetwork::costFunction(const std::vector<double>& aInput, size_t aCorrectDigit) {
    if (aInput.size() != 10) {
        throw std::logic_error("Cost Function error: Input vector should be size 10");
    }
    if (aCorrectDigit >= 10) {
        throw std::logic_error("Cost Function error: Correct expected digit should be less than 10");
    }
    std::vector<double> output(aInput.size());
    for (size_t i = 0; i < aInput.size(); ++i) {
        if (i == aCorrectDigit) {
            output[i] = -std::log(aInput[i]);
        }
        else {
            output[i] = -std::log(1 - aInput[i]);
        }
    }
    return output;
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

std::vector<double> NeuralNetwork::Softmax(const std::vector<double>& aInput) {
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
    aOut << "------Neural Network parameters------" << std::endl;
    aOut << "Size: " << aNeuralNetwork.size() << std::endl;
    aOut << "Epochs: " << aNeuralNetwork.mEpochs << std::endl;
    aOut << "Learning Rate: " << aNeuralNetwork.mLearningRate << std::endl << std::endl;
    for (size_t i = 0; i < aNeuralNetwork.size(); ++i) {
        aOut << "---Layer " << i + 1 << "---" << std::endl;
        aOut << "Nodes: " << aNeuralNetwork.mLayerOutputs[i].size() << std::endl;
        aOut << "Weights: " << std::endl;
        if (i < aNeuralNetwork.size() - 1) {
            //aOut << aNeuralNetwork.mWeights[i] << std::endl;
        }
        aOut << "Activation Function: " << aNeuralNetwork.activationFunctionName(i) << std::endl << std::endl;
    }
    return aOut;
}

