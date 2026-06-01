#include "../include/NeuralNetwork.h"
#include "../include/Util.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

// Constructors
NeuralNetwork::NeuralNetwork(std::vector<size_t>& aLayers) 
    : mGenerator{std::random_device()()}, mDist{-1.0, 1.0}, mEpochs{1}, mLearningRate{0.01} {
    if (aLayers.size() < 2) {
        throw std::invalid_argument("NeuralNetwork must be initialized with at least 2 layers");
    }

    mLayerOutputs.reserve(aLayers.size());
    mLayerOutputsTransformed.reserve(aLayers.size());
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
        mWeights.emplace_back(Mat2D<double>(aLayers[i], aLayers[i - 1], 1.0));
        randomizeMatrix(mWeights.back());
    }
    
    mActivationFunctions.resize(aLayers.size(), ActivationFunction::Sigmoid);

    // The last layer does not have an activation functions
    mActivationFunctions[aLayers.size() - 1] = ActivationFunction::None;   
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
    
    // Training finished
    std::cout << "------Training Finished------" << std::endl;
}

void NeuralNetwork::test(const std::vector<Mat2D<double>>& aImages, const std::vector<size_t>& aLabels) {
    if (aImages.size() != aLabels.size()) {
        std::string errorMessage = "Neural Network test failed: Number of input images and labels must match\n";
        errorMessage += std::string("Number of images: ") + std::to_string(aImages.size()) + "\n";
        errorMessage += "Number of labels: " + std::to_string(aLabels.size()) + "\n";
        throw std::invalid_argument(errorMessage);
    }

    size_t correctIterations = 0;
    for (size_t iteration = 0; iteration < aImages.size(); ++iteration) {
        std::cout << "------Testing Iteration " << std::to_string(iteration + 1) << "------" << std::endl;
        // Run current image through the network
        std::vector<double> output = forward(aImages[iteration].toVec());
       
        std::cout << "Loss: " << costFunction(output, aLabels[iteration]) << std::endl;

        // Compare network output with expected label
        size_t index = std::distance(output.begin(), std::max_element(output.begin(), output.end())) + 1;
        correctIterations += (index == aLabels[iteration]);
    }
    
    // Testing finished. Print output
    std::cout << "----Testing Finished----" << std::endl;
    std::cout << "Accuracy: " << std::to_string(correctIterations) << "/" << std::to_string(aImages.size()) << " -> ";
    std::cout << std::setprecision(2) << std::to_string(static_cast<double>(correctIterations) / aImages.size() * 100.0) << std::endl;
    
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

void NeuralNetwork::setTrainingData(const std::vector<Mat2D<double>>& aTrainingData) {
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

std::vector<double> NeuralNetwork::forward(const std::vector<double>& aInput) {
    // The first layer outputs the image input data 
    mLayerOutputs[0] = mLayerOutputsTransformed[0] = aInput;

    // Propagate input through each layer in the network, applying weights between each layer
	for (size_t weightsIdx = 0; weightsIdx < mWeights.size(); ++weightsIdx) {
        std::vector<double>& currOutput = mLayerOutputs[weightsIdx + 1]; 

        // Multiply weights matrix with output from previous layer
		currOutput = mWeights[weightsIdx].multiply(mLayerOutputsTransformed[weightsIdx]);

        // Save untransformed output to transformed vector to be transformed
        mLayerOutputsTransformed[weightsIdx + 1] = currOutput;

        // Retrieve Activation Function for current layer
        //const auto activationFunction = getActivationFunction(mActivationFunctions[weightsIdx]);

        // Apply activation function to each output in the current layer, except the last layer
        if (weightsIdx < mWeights.size() - 1) {
            for (double& element : mLayerOutputsTransformed[weightsIdx + 1]) {
                element = Sigmoid(element, false);
            }
        }
	}
    Softmax(mLayerOutputsTransformed.back());
    return mLayerOutputsTransformed.back(); 
}

void NeuralNetwork::backpropagation(size_t aCorrectDigit) {
    std::vector<double> expected(10, 0.0);
    expected[aCorrectDigit] = 1.0;
    std::vector<double> delta = Util::Subtract(mLayerOutputsTransformed.back(), expected); 
    for (int idx = mWeights.size() - 1; idx >= 0; --idx) {
        Mat2D<double> gradient = Util::VecToMatrix(delta, mLayerOutputsTransformed[idx]);

        auto oldWeights = mWeights[idx];

        // Gradient descent weight updates
        mWeights[idx] = mWeights[idx] - gradient.scalar(mLearningRate);

        if (idx > 0) {
            delta = oldWeights.transpose().multiply(delta);
            auto currLayerOutputsTransformed = mLayerOutputsTransformed[idx];

            const auto activationFunction = getActivationFunction(mActivationFunctions[idx]);
            if (activationFunction) {
                for (double& element : currLayerOutputsTransformed) {
                    element = activationFunction(element, true); 
                }
            }
            delta = Util::multiply(delta, currLayerOutputsTransformed);
        }
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

double NeuralNetwork::costFunction(const std::vector<double>& aInput, size_t aCorrectDigit) {
    if (aInput.size() != 10) {
        throw std::logic_error("Cost Function error: Input vector should be size 10");
    }
    if (aCorrectDigit >= 10) {
        throw std::logic_error("Cost Function error: Correct expected digit should be less than 10");
    }
    constexpr double eps = 1e-15;
    return -std::log(aInput[aCorrectDigit] + eps);
}

double NeuralNetwork::Sigmoid(double aInput, bool aDerivative) {
    aInput = std::clamp(aInput, -50.0, 50.0);
    if (aDerivative) {
        // Sigmoid allows for a simple trick here, assuming aInput is the
        // activation values.
        return aInput * (1.0 - aInput);
    }
	return 1.0 / (1.0 + std::exp(-aInput));
}

double NeuralNetwork::ReLu(double aInput, bool aDerivative) {
    if (aDerivative) {
        return aInput;
    }
	return std::max(0.0, aInput);
}

void NeuralNetwork::Softmax(std::vector<double>& aInput) {
    double maxElement = *max_element(aInput.begin(), aInput.end());
    double sum = 0;
    
    // Calculate exponentiated of input vector.
    for (size_t i = 0; i < aInput.size(); ++i) {
        aInput[i] = std::exp(aInput[i] - maxElement);
        sum += aInput[i]; 
    }

    // Divide individual exponentiated values by exponentiated sum to bind
    // sum of output to 1.
    std::for_each(aInput.begin(), aInput.end(), [sum](double& element) {
        element /= sum;
    });
}

void NeuralNetwork::randomizeMatrix(Mat2D<double>& aMat) {
    for (size_t i = 0; i < aMat.height(); ++i) {
        for (size_t j = 0; j < aMat.width(); ++j) {
            aMat[i][j] = mDist(mGenerator);
        }
    }
}

std::ostream& operator<<(std::ostream& aOut, NeuralNetwork& aNeuralNetwork) {
    aOut << "------Neural Network parameters------" << std::endl;
    aOut << "Number of Layers: " << aNeuralNetwork.size() << std::endl;
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
    std::cout << "---------------------------------------------" << std::endl;
    return aOut;
}

