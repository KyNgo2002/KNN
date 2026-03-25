#include "Node.h"

Node::Node() {
    mNext = nullptr;    
}

void Node::setNext(Node* nextNode) {
    if (mNext) 
        mNext->mNext = nextNode;
}
