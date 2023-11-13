//
// Created by BigO on 2020/11/19.
//

#ifndef CNN_LAYER_H
#define CNN_LAYER_H
#pragma once
#include <vector>
#include "Neuron.h"

class Layer {
public:
    Layer() {}
    ~Layer() {}

    void printLayer() const {}

public:
    std::vector<Neuron> listOfNeurons;
    size_t numberOfNeuronInLayer;
};


#endif //CNN_LAYER_H
