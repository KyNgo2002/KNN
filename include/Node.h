#ifndef NODE_H
#define NODE_H
    
class Node {
public:
    Node();
    void setNext(Node* nextNode);    
private:
    Node* mNext;
};

#endif
