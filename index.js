import * as tf from '@tensorflow/tfjs'
import {twoDimNeuralNetwork} from "./sketches/2dNNVisualization";

console.log('index.js working')

// Solve for XOR
const LEARNING_RATE = 0.01;
const EPOCHS = 200;

let model

function buildModel() {
    // Define the model.
    const model = tf.sequential();
    // Set up the network tf.layers
    model.add(tf.layers.dense({units: 8, activation: 'relu', inputShape: [2], useBias: true}));
    model.add(tf.layers.dense({units: 16, activation: 'relu', useBias: true}));
    model.add(tf.layers.dense({units: 32, activation: 'relu', useBias: true}));
    model.add(tf.layers.dense({units: 16, activation: 'relu', useBias: true}));
    model.add(tf.layers.dense({units: 8, activation: 'relu', useBias: true}));
    model.add(tf.layers.dense({units: 2, useBias: true}));
    // Define the optimizer
    const optimizer = tf.train.adam(LEARNING_RATE);
    // Init the model
    model.compile({
        optimizer: optimizer,
        loss: tf.losses.sigmoidCrossEntropy,
        metrics: ['accuracy'],

    });
    return model
}

twoDimNeuralNetwork(500, buildModel())

