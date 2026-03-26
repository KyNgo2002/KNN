#ifndef LAYER_H
#define LAYER_H

#include "Node.h"

#include <vector>

class Layer {
public:
    Layer() = default;
    Layer(unsigned int aLayerSize);

    size_t size() const { return mNodes.size(); }
private:
    std::vector<Node> mNodes;
};

#endif
