//
// Created by BigO on 2020/11/19.
//

#ifndef C___NEURON_H
#define C___NEURON_H
#pragma once
#include <stdlib.h>
#include <vector>

class Neuron {
public:
    Neuron(){}
    ~Neuron(){}  // 소멸자 ~

    double initNeuron() { return ((double)rand()) / RAND_MAX; }

public:
    std::vector<double> listOfWeightIn;
    std::vector<double> listOfWeightOut;
    double outputValue;
    double error;
    double sensibility;
};


#endif //C___NEURON_H
