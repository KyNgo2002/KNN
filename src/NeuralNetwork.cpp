#include "../include/NeuralNetwork.h"
#include "../include/Util.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>

// Constructors
NeuralNetwork::NeuralNetwork(std::vector<size_t>& aLayers) 
    : mGenerator{std::random_device()()}, mEpochs{1}, mLearningRate{0.01} {
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
    mBiases.reserve(aLayers.size() - 1);
    for (size_t i = 1; i < aLayers.size(); ++i) {
        mWeights.emplace_back(Mat2D<double>(aLayers[i], aLayers[i - 1], 1.0));
        double boundary = std::sqrt(2.0 / aLayers[i - 1]);
        mDist = std::normal_distribution<double>(0.0, boundary);
        randomizeMatrix(mWeights.back());
        mBiases.emplace_back(aLayers[i], 0.01);
    }
    
    mActivationFunctions.resize(aLayers.size(), ActivationFunction::ReLu);

    mActivationFunctions[mActivationFunctions.size() - 1] = ActivationFunction::None;
}

NeuralNetwork::NeuralNetwork(const std::string& aModelFilePath) {
    std::cout << "Attempting to create model from model file path: " << aModelFilePath;
}

NeuralNetwork::NeuralNetwork(size_t aEpochs, double aLearningRate, const std::vector<Mat2D<double>>& aWeights, const std::vector<std::vector<double>>& aBiases, const std::vector<ActivationFunction>& aActivationFunctions) {
    if (aWeights.size() != aBiases.size()) {
        std::string errorMessage = "Creation of Neural Network failed. Number of weights matrices and biases vectors must match.\n";
        errorMessage += "Number of weights matrices: " + std::to_string(aWeights.size()) + "\n";
        errorMessage += "Number of bias vectors: " + std::to_string(aBiases.size()) + "\n";
        throw std::invalid_argument(errorMessage);
    }

    if (aActivationFunctions.size() != aWeights.size()) {
        std::string errorMessage = "Creation of Neural Network failed. Number of activation functions must match the number of weights/bias containers.\n";
        errorMessage += "Expected number of activation functions: " + std::to_string(aBiases.size()) + "\n";
        errorMessage += "Number of provided activation functions: " + std::to_string(aActivationFunctions.size()) + "\n";
    }

    // Verify weights matrices
    if (aWeights[0].width() != 784) {
        throw std::length_error("Construction of Neural Network failed. First matrix must have a width of 784.");
    }
    if (aWeights.back().height() != 10) {
        throw std::length_error("Construction of Neural Network failed. Last matrix must ahve a height of 10.");
    }

    for (size_t i = 0; i < aWeights.size() - 1; ++i) {
        if (aWeights[i].height() != aWeights[i + 1].width()) {
            std::string errorMessage = "Construction of Neural Network failed. Incorrect size of consecutive matrices\n";
            errorMessage += "Matrix " + std::to_string(i + 1) + " size: " + aWeights[i].size() + "\n";
            errorMessage += "Matrix " + std::to_string(i + 2) + " size: " + aWeights[i + 1].size() + "\n";
            throw std::invalid_argument(errorMessage);
        }
    }

    // Verify bias vectors
    for (size_t i = 0; i < aBiases.size(); ++i) {
        if (aBiases[i].size() != aWeights[i].height()) {
            std::string errorMessage = "Construction of Neural Network failed. Incorrect size of bias vectors.\n";
            errorMessage += "Expected size: " + std::to_string(aBiases[i].size()) + "\n";
            errorMessage += "Bias Vector " + std::to_string(i + 1) + " size: " + std::to_string(aBiases[i].size()) + "\n";
            throw std::invalid_argument(errorMessage);
        }
    }
    
    // Initial verification successful, create network
    mEpochs = aEpochs;
    mLearningRate = aLearningRate;
    mWeights = aWeights;
    mBiases = aBiases;
    mActivationFunctions = aActivationFunctions;
    mLayerOutputs.resize(aWeights.size() + 1);
    mLayerOutputsTransformed.resize(aWeights.size() + 1);
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

std::string NeuralNetwork::getActivationFunctionName(size_t ind) const { 
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

void NeuralNetwork::test() {
    test(mTestingData.size());
}

void NeuralNetwork::test(size_t aIterations) {
    if (mTestingData.size() == 0 || mTestingLabels.size() == 0) {
        std::string errorMessage = "Neural Network unable to test: Testing data and labels must be provided.\n";
        errorMessage += "Provided test data: " + std::to_string(mTestingData.size()) + "\n";
        errorMessage += "Provided number of labels: " + std::to_string(mTrainingLabels.size()) + "\n";
        throw std::invalid_argument(errorMessage);
    }
    if (mTestingData.size() != mTestingLabels.size()) {
        std::string errorMessage = "Neural Network test failed: Number of input images and labels must match\n";
        errorMessage += "Number of images: " + std::to_string(mTestingData.size()) + "\n";
        errorMessage += "Number of labels: " + std::to_string(mTrainingLabels.size()) + "\n";
        throw std::invalid_argument(errorMessage);
    }

    size_t correctIterations = 0;
    for (size_t iteration = 0; iteration < std::min(mTestingData.size(), aIterations); ++iteration) {
        std::cout << "------Testing Iteration " << std::to_string(iteration + 1) << "------" << std::endl;
        // Run current image through the network
        std::vector<double> output = forward(mTestingData[iteration].toVec());
        // Compare network output with expected label
        size_t index = std::distance(output.begin(), std::max_element(output.begin(), output.end())) ;
        correctIterations += (index == mTestingLabels[iteration]);
    }
    
    // Testing finished. Print output
    std::cout << "----Testing Finished----" << std::endl;
    std::cout << "Accuracy: " << std::to_string(correctIterations) << "/" << std::to_string(mTestingData.size()) << " -> ";
    std::cout << std::setprecision(2) << std::to_string(static_cast<double>(correctIterations) / mTestingData.size() * 100.0) << std::endl;
}

NeuralNetwork NeuralNetwork::readModel(const std::string& aModelFilePath) {
    std::cout << "Attempting to read model at file path: " << aModelFilePath << std::endl;
    std::ifstream inFile(aModelFilePath);
    
    if (!inFile.is_open()) {
        std::cout << "Failed to open model file path for reading." << std::endl;
        return NeuralNetwork();
    }
    std::string line;

    auto parseLine = [](const std::string& aStr) -> std::string {
        size_t i = aStr.size() - 1;
        for (; i > 0; --i) {
            if (aStr[i] == ' ') {
                return aStr.substr(i + 1);
            }
        }
        return "";
    };

    auto getActivationFunctionHelper = [](const std::string& aActivationFunctionName) -> ActivationFunction {
        if (aActivationFunctionName == "Sigmoid") {
            return ActivationFunction::Sigmoid;
        }
        else if (aActivationFunctionName == "ReLu") {
            return ActivationFunction::ReLu;
        }
        else {
            return ActivationFunction::None;
        }
    };

    // Epochs
    getline(inFile, line);
    size_t numEpochs = std::stoul(parseLine(line));

    // Learning rate
    getline(inFile, line);
    double learningRate = std::stod(parseLine(line));

    // Number of layers
    getline(inFile, line);
    size_t numLayers = std::stoul(parseLine(line));

    std::vector<Mat2D<double>> weights;
    std::vector<std::vector<double>> biases;
    std::vector<ActivationFunction> activationFunctions;
    // Read per-layer information
    for (size_t layer = 0; layer < numLayers; ++layer) {
        // Layer number line
        getline(inFile, line);        

        // Activation Function
        getline(inFile, line);
        std::string activationFunction = parseLine(line);
        activationFunctions.push_back(getActivationFunctionHelper(activationFunction));

        // Layer size
        getline(inFile, line);
        size_t layerSize = std::stoul(parseLine(line));

        if (layer < numLayers - 1) {
            // Next Layer size
            getline(inFile, line);
            size_t nextLayerSize = std::stoul(parseLine(line));

            // Weights burner word
            getline(inFile, line);

            // Weights Matrix 
            Mat2D<double> weightsMatrix(nextLayerSize, layerSize);
            for (size_t row = 0; row < nextLayerSize; ++row) {
                std::string token;
                //Grab entire row in matrix
                getline(inFile, line);
                std::istringstream ss(line);
                for (size_t col = 0; col < layerSize; ++col) {
                    ss >> token;
                    weightsMatrix[row][col] = std::stod(token);
                }
            }
            weights.push_back(weightsMatrix);

            // Biases burner word
            getline(inFile, line);

            // Biases Matrix 
            std::vector<double> bias(nextLayerSize);
            getline(inFile, line);
            std::string token;
            std::stringstream ss(line);
            for (size_t i = 0; i < nextLayerSize; ++i) {
                ss >> token;
                bias[i] = std::stod(token);
            }
            biases.push_back(bias);
        }
    }

    std::cout << "Successfully read model from file at path: " << aModelFilePath << std::endl;
    inFile.close();

    // Create new model
    NeuralNetwork network(numEpochs, learningRate, weights, biases, activationFunctions);

    return network;
}

void NeuralNetwork::writeModel(const std::string& aModelFilePath) {
    std::cout << "Attempting to write model at file path: " << aModelFilePath << std::endl;
    std::ofstream outFile(aModelFilePath);

    if (!outFile.is_open()) {
        std::cout << "Failed to open model file path for writing." << std::endl;
        return;
    }

    // Write model parameters to output file
    outFile << "Epochs: " << mEpochs << std::endl;
    outFile << "Learning Rate: " << mLearningRate << std::endl;
    outFile << "Number of Layers: " << mLayerOutputs.size() << std::endl;
    for (size_t i = 0; i < mLayerOutputs.size(); ++i) {
        outFile << "---Layer " << i + 1 << "---" << std::endl;
        outFile << "Activation Function: " << getActivationFunctionName(i) << std::endl;
        outFile << "Layer size: " << mLayerOutputs[i].size() << std::endl;
        if (i < mLayerOutputs.size() - 1) {
            outFile << "Next Layer Size: " << mLayerOutputs[i + 1].size() << std::endl;
            // Write weights matrices
            outFile << "Weights:" << std::endl;
            for (size_t row = 0; row < mWeights[i].height(); ++row) {
                for (const auto& element : mWeights[i][row]) {
                    outFile << element << " ";
                }
                outFile << std::endl;
            }
            // Write bias matrices
            outFile << "Biases: " << std::endl;
            for (const auto& element : mBiases[i]) {
                outFile << element << " ";
            }
            outFile << std::endl;
        }
    }
    std::cout << "Successfully wrote model to file at path: " << aModelFilePath << std::endl;
    outFile.close();
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

void NeuralNetwork::setTestingData(const std::vector<Mat2D<double>>& aTestingData) {
    mTestingData = aTestingData;
}

void NeuralNetwork::setTestingLabels(const std::vector<size_t>& aTestingLabels) {
    mTestingLabels = aTestingLabels;
}

void NeuralNetwork::setActivationFunction(size_t ind, ActivationFunction aFunction) {
    if (ind >= mLayerOutputs.size()) {
        throw std::out_of_range("Invalid index provided to set activation function");
    }
    if (ind == 0 || ind == mActivationFunctions.size() - 1) {
        throw std::invalid_argument("Cannot set the activation function for the first or last layer");
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

        // Add biases
        for (size_t i = 0; i < currOutput.size(); ++i) {
            currOutput[i] += mBiases[weightsIdx][i];
        }

        // Save untransformed output to transformed vector to be transformed
        mLayerOutputsTransformed[weightsIdx + 1] = currOutput;

        // Apply activation function to each output in the current layer, except the last layer
        if (weightsIdx < mWeights.size() - 1) {
            for (double& element : mLayerOutputsTransformed[weightsIdx + 1]) {
                element = ReLu(element, false);
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
        mWeights[idx] = mWeights[idx] - gradient.scalar(mLearningRate);
        for (size_t i = 0; i < delta.size(); ++i) {
            mBiases[idx][i] -= mLearningRate * delta[i];
        }

        if (idx > 0) {
            delta = oldWeights.transpose().multiply(delta);
            std::vector<double> derivative = mLayerOutputs[idx];

            for (double& element : derivative) {
                element = ReLu(element, true); 
            }
            delta = Util::multiply(delta, derivative);
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
        return aInput > 0 ? 1 : 0;
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
        aOut << "Activation Function: " << aNeuralNetwork.getActivationFunctionName(i) << std::endl << std::endl;
    }
    std::cout << "---------------------------------------------" << std::endl;
    return aOut;
}

