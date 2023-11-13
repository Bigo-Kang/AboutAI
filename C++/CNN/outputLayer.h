// https://github.com/dani2442/NeuralNetwork_Tutorial/
// Created by BigO on 2020/11/19.
//

#ifndef CNN_OUTPUTLAYER_H
#define CNN_OUTPUTLAYER_H
#pragma once
#include <iostream>
#include "Layer.h"

class outputLayer: public Layer {
public:
    outputLayer(){};
    ~outputLayer(){};

    outputLayer &initLayer(outputLayer &outputLayer);
    void printLayer(outputLayer& outputLayer);

private:

};

inline outputLayer &outputLayer::initLayer(outputLayer &outputLayer) {
    std::vector<double> listOfWeightOutTmp;
    std::vector<Neuron> listOfNeurons;

    for (size_t i = 0; i < outputLayer.numberOfNeuronInLayer; i++) {
        Neuron neuron;
        listOfWeightOutTmp.push_back(neuron.initNeuron());

        neuron.listOfWeightIn = listOfWeightOutTmp;
        listOfNeurons.push_back(neuron);

        listOfWeightOutTmp.clear();
    }
    outputLayer.listOfNeurons = listOfNeurons;
    return outputLayer;
}

inline void outputLayer::printLayer(outputLayer &outputLayer) {
    std::cout << "### OUTPUT LAYER ###" << std::endl;
    int n = 1;
    for (Neuron &neuron : outputLayer.listOfNeurons) {
        std::cout << "Neuron #" << n << ":" << std::endl;
        std::cout << "Output Weight : " << std::endl;
        std::vector<double> weights = neuron.listOfWeightOut;
        for (double weight : weights) {
            std::cout << weight << std::endl;
        }
        n++;
    }
}


#endif //CNN_OUTPUTLAYER_H

