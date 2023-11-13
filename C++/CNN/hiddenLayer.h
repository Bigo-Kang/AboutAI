// https://github.com/dani2442/NeuralNetwork_Tutorial/
// Created by BigO on 2020/11/19.
//

#ifndef CNN_HIDDENLAYER_H
#define CNN_HIDDENLAYER_H
#pragma once
#include "Layer.h"
#include "inputLayer.h"
#include "outputLayer.h"

class hiddenLayer: public Layer {
public:
    hiddenLayer() {}
    ~hiddenLayer() {}

    std::vector<hiddenLayer> &initLayer(const  hiddenLayer&, std::vector<hiddenLayer>&, const inputLayer&, const outputLayer&);
    void printLayer(const std::vector<hiddenLayer>& listOfHiddenLayer) const;

private:

};

inline std::vector<hiddenLayer>& hiddenLayer::initLayer(
        const hiddenLayer & hiddenLayer,
        std::vector<hiddenLayer>& listOfHiddenLayer,
        const inputLayer & inputLayer,
        const outputLayer & outputLayer)
{
    std::vector<double> listOfWeightIn;
    std::vector<double> listOfWeightOut;
    std::vector<Neuron> listOfNeurons;

    size_t numberOfHiddenLayers = listOfHiddenLayer.size();

    for (size_t i = 0; i < numberOfHiddenLayers; i++){
        for (size_t j = 0; j < hiddenLayer.numberOfNeuronInLayer; j++) {
            Neuron neuron;

            size_t limitIn;
            size_t limitOut;

            if (i == 0){
                limitIn = inputLayer.numberOfNeuronInLayer;
                if (numberOfHiddenLayers > 1)
                    limitOut = listOfHiddenLayer[i+1].numberOfNeuronInLayer;
                else if (numberOfHiddenLayers == 1)
                    limitOut = outputLayer.numberOfNeuronInLayer;
            }
            else if (i == numberOfHiddenLayers-1) {
                limitIn = listOfHiddenLayer[i-1].numberOfNeuronInLayer;
                limitOut = outputLayer.numberOfNeuronInLayer;
            }
            else {
                limitIn = listOfHiddenLayer[i-1].numberOfNeuronInLayer;
                limitOut = listOfHiddenLayer[i+1].numberOfNeuronInLayer;
            }
            // Bias in not connected
            limitIn--;
            limitOut--;

            if (j >= 1) {
                for (size_t k = 0; k <= limitIn; k++)
                    listOfWeightIn.push_back(neuron.initNeuron());
            }
            for (size_t k = 0; k <= limitOut; k++)
                listOfWeightOut.push_back(neuron.initNeuron());

            neuron.listOfWeightIn = listOfWeightIn;
            neuron.listOfWeightOut = listOfWeightOut;
            listOfNeurons.push_back(neuron);

            listOfWeightIn.clear();
            listOfWeightOut.clear();
        }
        listOfHiddenLayer[i].listOfNeurons = listOfNeurons;
        this->listOfNeurons = listOfNeurons;
        listOfNeurons.clear();
    }
    return listOfHiddenLayer;
}

inline void hiddenLayer::printLayer(const std::vector<hiddenLayer> &listOfHiddenLayer) const {
    std::cout << "### HIDDEN LAYER ###" << std::endl;
    int h = 1;
    for (hiddenLayer hiddenLayer1 : listOfHiddenLayer) {
        std::cout << "Hidden Layer #" << h << std::endl;
        int n = 1;
        for (Neuron& neuron : hiddenLayer1.listOfNeurons) {
            std::cout << "Neuron #" << n << std::endl;
            std::cout << "Input Weights : " << std::endl;
            std::vector<double> weights = neuron.listOfWeightIn;
            for (double weight : weights)
                std::cout << weight << " ";
            std::cout << std::endl << "Output Weights : " << std::endl;
            weights = neuron.listOfWeightOut;
            for (double weight : weights)
                std::cout << weight << " ";
            std::cout << std::endl;
            n++;
        }
        h++;
    }
}

#endif //CNN_HIDDENLAYER_H
