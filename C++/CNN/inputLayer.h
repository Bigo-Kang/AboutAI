// https://github.com/dani2442/NeuralNetwork_Tutorial/
// Created by BigO on 2020/11/19.
//

#ifndef CNN_INPUTLAYER_H
#define CNN_INPUTLAYER_H
#pragma once
#include <iostream>
#include "Layer.h"

class inputLayer: public Layer {
public:
    inputLayer(){};
    ~inputLayer(){};

    inputLayer &initLayer(inputLayer &inputLayer);
    void printLayer(inputLayer& inputLayer);

private:

};

inline inputLayer &inputLayer::initLayer(inputLayer &inputLayer) {
    std::vector<double> listOfWeightInTmp;
    std::vector<Neuron> listOfNeurons;

    for (size_t i = 0; i < inputLayer.numberOfNeuronInLayer; i++) {
        Neuron neuron;
        listOfWeightInTmp.push_back(neuron.initNeuron());

        neuron.listOfWeightIn = listOfWeightInTmp;
        listOfNeurons.push_back(neuron);

        listOfWeightInTmp.clear();
    }
    inputLayer.listOfNeurons = listOfNeurons;
    return inputLayer;
}

//inline is calling function
inline void inputLayer::printLayer(inputLayer &inputLayer) {
    std::cout << "### INPUT LAYER ###" << std::endl;
    int n = 1;
    for (Neuron &neuron : inputLayer.listOfNeurons) {
        std::cout << "Neuron #" << n << ":" << std::endl;
        std::cout << "Input Weight : " << std::endl;
        std::vector<double> weights = neuron.listOfWeightIn;
        for (double weight : weights) {
            std::cout << weight << std::endl;
        }
        n++;
    }
}


#endif //CNN_INPUTLAYER_H
