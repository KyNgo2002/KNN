#ifndef LAYER_H
#define LAYER_H

#include "Node.h"
#include "Mat2D.h"

#include <vector>

template <typename T>
class Layer {
public:
    Layer() = default;
    Layer(unsigned int aLayerSize);

    size_t size() const { return mNodes.size(); }

private:
    std::vector<Node> mNodes;
    std::vector<Mat2D<T>> mWeights;
};

template <typename T>
Layer<T>::Layer(unsigned int aLayerSize) {
    mNodes.resize(aLayerSize);
}

#endif
