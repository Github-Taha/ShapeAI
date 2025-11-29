class NeuralNetworkv2 {
    constructor (inputLayerCount, hiddenLayerCount, outputLayerCount) {
        this.inputLayerCount = inputLayerCount;
        this.hiddenLayerCount = hiddenLayerCount;
        this.outputLayerCount = outputLayerCount;

        // Hidden Layer
        this.hiddenLayerWeights = new Float32Array(inputLayerCount * hiddenLayerCount).map((v) => Math.random() * 2 - 1);
        this.hiddenLayerBiases = new Float32Array(hiddenLayerCount).map((v) => Math.random() * 2 - 1);

        // Output Layer
        this.outputLayerWeights = new Float32Array(hiddenLayerCount * outputLayerCount).map((v) => Math.random() * 2 - 1);
        this.outputLayerBiases = new Float32Array(outputLayerCount).map((v) => Math.random() * 2 - 1);

        this.learningRate = 0.01;
    }

    feedForward (inputs) {
        if (inputs.length != this.inputLayerCount)
            return null;

        let hiddenValues = new Float32Array(this.hiddenLayerCount).fill(0);
        let outputValues = new Float32Array(this.outputLayerCount).fill(0);

        for (let i = 0; i < this.hiddenLayerCount; i++) {
            // For each output neuron
            for (let j = 0; j < this.inputLayerCount; j++)
                hiddenValues[i] += inputs[j] * this.hiddenLayerWeights[this.inputLayerCount * i + j];
            hiddenValues[i] += this.hiddenLayerBiases[i];
            hiddenValues[i] = sigmoid(hiddenValues[i]);
        }

        for (let i = 0; i < this.outputLayerCount; i++) {
            // For each output neuron
            for (let j = 0; j < this.hiddenLayerCount; j++)
                outputValues[i] += hiddenValues[j] * this.outputLayerWeights[this.hiddenLayerCount * i + j];
            outputValues[i] += this.outputLayerBiases[i];
            outputValues[i] = sigmoid(outputValues[i]);
        }

        return outputValues;
    }

    backPropagation(inputs, targets) {
        if (inputs.length !== this.inputLayerCount) return false;
        if (targets.length !== this.outputLayerCount) return false;

        // Forward pass (raw values and activations)
        let hiddenRaw = new Float32Array(this.hiddenLayerCount);
        let hidden = new Float32Array(this.hiddenLayerCount);
        for (let i = 0; i < this.hiddenLayerCount; i++) {
            let sum = 0;
            for (let j = 0; j < this.inputLayerCount; j++)
                sum += inputs[j] * this.hiddenLayerWeights[i * this.inputLayerCount + j];
            sum += this.hiddenLayerBiases[i];
            hiddenRaw[i] = sum;
            hidden[i] = sigmoid(sum);
        }

        let outRaw = new Float32Array(this.outputLayerCount);
        let out = new Float32Array(this.outputLayerCount);
        for (let i = 0; i < this.outputLayerCount; i++) {
            let sum = 0;
            for (let j = 0; j < this.hiddenLayerCount; j++)
                sum += hidden[j] * this.outputLayerWeights[i * this.hiddenLayerCount + j];
            sum += this.outputLayerBiases[i];
            outRaw[i] = sum;
            out[i] = sigmoid(sum);
        }

        // Output deltas
        let deltaOut = new Float32Array(this.outputLayerCount);
        for (let i = 0; i < this.outputLayerCount; i++) {
            let error = targets[i] - out[i];
            deltaOut[i] = error * out[i] * (1 - out[i]); // derivative of sigmoid
        }

        // Hidden deltas
        let deltaHidden = new Float32Array(this.hiddenLayerCount);
        // hidden_error[j] = sum_i (deltaOut[i] * weight_i_j)
        for (let j = 0; j < this.hiddenLayerCount; j++) {
            let sum = 0;
            for (let i = 0; i < this.outputLayerCount; i++) {
                sum += deltaOut[i] * this.outputLayerWeights[i * this.hiddenLayerCount + j];
            }
            deltaHidden[j] = sum * hidden[j] * (1 - hidden[j]); // derivative on hidden
        }

        // Apply updates for output weights and biases
        for (let i = 0; i < this.outputLayerCount; i++) {
            for (let j = 0; j < this.hiddenLayerCount; j++) {
                let grad = deltaOut[i] * hidden[j]; // gradient for weight i <-- j
                this.outputLayerWeights[i * this.hiddenLayerCount + j] += this.learningRate * grad;
            }
            this.outputLayerBiases[i] += this.learningRate * deltaOut[i];
        }

        // Apply updates for hidden weights and biases
        for (let j = 0; j < this.hiddenLayerCount; j++) {
            for (let k = 0; k < this.inputLayerCount; k++) {
                let grad = deltaHidden[j] * inputs[k];
                this.hiddenLayerWeights[j * this.inputLayerCount + k] += this.learningRate * grad;
            }
            this.hiddenLayerBiases[j] += this.learningRate * deltaHidden[j];
        }

        return true;
    }
}

class NeuralNetwork2v2 {
    constructor (inputLayerCount, hiddenLayersCount, outputLayerCount) {
        this.inputLayerCount = inputLayerCount;
        this.hiddenLayersCount = hiddenLayersCount;
        this.outputLayerCount = outputLayerCount;

        // Hidden Layers
        this.hiddenLayersWeights = new Array(this.hiddenLayersCount.length);
        for (let i = 0; i < this.hiddenLayersCount.length; i++) {
            this.hiddenLayersWeights[i] = new Float32Array(
                (i === 0 ? this.inputLayerCount : this.hiddenLayersCount[i - 1]) * this.hiddenLayersCount[i]
            ).map((v) => Math.random() * 2 - 1);
        }
        this.hiddenLayersBiases = new Array(this.hiddenLayersCount.length);
        for (let i = 0; i < this.hiddenLayersCount.length; i++) {
            this.hiddenLayersBiases[i] = new Float32Array(this.hiddenLayersCount[i]).map((v) => Math.random() * 2 - 1);
        }

        // Output Layer
        this.outputLayerWeights = new Float32Array(this.hiddenLayersCount[this.hiddenLayersCount.length - 1] * this.outputLayerCount).map((v) => Math.random() * 2 - 1);
        this.outputLayerBiases = new Float32Array(this.outputLayerCount).map((v) => Math.random() * 2 - 1);

        this.learningRate = 0.01;
    }

    feedForward (inputs) {
        if (inputs.length != this.inputLayerCount)
            return null;

        let hiddenLayersValues = new Array(this.hiddenLayersCount.length);
        for (let i = 0; i < this.hiddenLayersCount.length; i++) {
            hiddenLayersValues[i] = new Float32Array(this.hiddenLayersCount[i]).fill(0);
        }
        let outputLayerValues = new Float32Array(this.outputLayerCount).fill(0);

        for (let a = 0; a < this.hiddenLayersCount.length; a++) {
            for (let i = 0; i < this.hiddenLayersCount[a]; i++) {
                // For each output neuron
                const count = a === 0 ? this.inputLayerCount : this.hiddenLayersCount[a - 1];
                for (let j = 0; j < count; j++) {
                    hiddenLayersValues[a][i] += 
                        (a === 0 ? inputs[j] : hiddenLayersValues[a - 1][j]) * 
                        this.hiddenLayersWeights[a][i * count + j];
                }
                hiddenLayersValues[a][i] += this.hiddenLayersBiases[a][i];
                hiddenLayersValues[a][i] = sigmoid(hiddenLayersValues[a][i]);
            }
        }

        for (let i = 0; i < this.outputLayerCount; i++) {
            // For each output neuron
            for (let j = 0; j < this.hiddenLayersCount[this.hiddenLayersCount.length - 1]; j++)
                outputLayerValues[i] += hiddenLayersValues[this.hiddenLayersCount.length - 1][j] * this.outputLayerWeights[i * this.hiddenLayersCount[this.hiddenLayersCount.length - 1] + j];
            outputLayerValues[i] += this.outputLayerBiases[i];
            outputLayerValues[i] = sigmoid(outputLayerValues[i]);
        }

        return outputLayerValues;
    }

    backPropagation(inputs, targets) {
        if (inputs.length !== this.inputLayerCount) return false;
        if (targets.length !== this.outputLayerCount) return false;

        // Forward pass: store raw and activated values per hidden layer
        let hiddenRaw = new Array(this.hiddenLayersCount.length);
        let hiddenAct = new Array(this.hiddenLayersCount.length);
        for (let a = 0; a < this.hiddenLayersCount.length; a++) {
            hiddenRaw[a] = new Float32Array(this.hiddenLayersCount[a]);
            hiddenAct[a] = new Float32Array(this.hiddenLayersCount[a]);
            const prevCount = (a === 0 ? this.inputLayerCount : this.hiddenLayersCount[a - 1]);
            for (let i = 0; i < this.hiddenLayersCount[a]; i++) {
                let sum = 0;
                for (let j = 0; j < prevCount; j++) {
                    let prevVal = (a === 0 ? inputs[j] : hiddenAct[a - 1][j]);
                    sum += prevVal * this.hiddenLayersWeights[a][i * prevCount + j];
                }
                sum += this.hiddenLayersBiases[a][i];
                hiddenRaw[a][i] = sum;
                hiddenAct[a][i] = sigmoid(sum);
            }
        }

        // Output layer forward
        let outRaw = new Float32Array(this.outputLayerCount);
        let outAct = new Float32Array(this.outputLayerCount);
        const lastHiddenCount = this.hiddenLayersCount[this.hiddenLayersCount.length - 1];
        for (let i = 0; i < this.outputLayerCount; i++) {
            let sum = 0;
            for (let j = 0; j < lastHiddenCount; j++)
                sum += hiddenAct[this.hiddenLayersCount.length - 1][j] * this.outputLayerWeights[i * lastHiddenCount + j];
            sum += this.outputLayerBiases[i];
            outRaw[i] = sum;
            outAct[i] = sigmoid(sum);
        }

        // Output deltas
        let deltaOut = new Float32Array(this.outputLayerCount);
        for (let i = 0; i < this.outputLayerCount; i++) {
            let err = targets[i] - outAct[i];
            deltaOut[i] = err * outAct[i] * (1 - outAct[i]);
        }

        // Backpropagate deltas into hidden layers
        let deltasHidden = new Array(this.hiddenLayersCount.length);
        for (let a = 0; a < this.hiddenLayersCount.length; a++)
            deltasHidden[a] = new Float32Array(this.hiddenLayersCount[a]);

        // Last hidden layer error from output
        const lastIdx = this.hiddenLayersCount.length - 1;
        for (let j = 0; j < this.hiddenLayersCount[lastIdx]; j++) {
            let sum = 0;
            for (let i = 0; i < this.outputLayerCount; i++)
                sum += deltaOut[i] * this.outputLayerWeights[i * this.hiddenLayersCount[lastIdx] + j];
            deltasHidden[lastIdx][j] = sum * hiddenAct[lastIdx][j] * (1 - hiddenAct[lastIdx][j]);
        }

        // Propagate backward through hidden layers
        for (let a = this.hiddenLayersCount.length - 1; a > 0; a--) {
            const curCount = this.hiddenLayersCount[a];
            const prevCount = this.hiddenLayersCount[a - 1];
            for (let j = 0; j < prevCount; j++) {
                let sum = 0;
                for (let i = 0; i < curCount; i++) {
                    sum += deltasHidden[a][i] * this.hiddenLayersWeights[a][i * prevCount + j];
                }
                deltasHidden[a - 1][j] = sum * hiddenAct[a - 1][j] * (1 - hiddenAct[a - 1][j]);
            }
        }

        // Update output weights & biases
        for (let i = 0; i < this.outputLayerCount; i++) {
            for (let j = 0; j < lastHiddenCount; j++) {
                let grad = deltaOut[i] * hiddenAct[lastIdx][j];
                this.outputLayerWeights[i * lastHiddenCount + j] += this.learningRate * grad;
            }
            this.outputLayerBiases[i] += this.learningRate * deltaOut[i];
        }

        // Update hidden weights & biases
        for (let a = 0; a < this.hiddenLayersCount.length; a++) {
            const prevCount = (a === 0 ? this.inputLayerCount : this.hiddenLayersCount[a - 1]);
            for (let i = 0; i < this.hiddenLayersCount[a]; i++) {
                for (let j = 0; j < prevCount; j++) {
                    const prevVal = (a === 0 ? inputs[j] : hiddenAct[a - 1][j]);
                    let grad = deltasHidden[a][i] * prevVal;
                    this.hiddenLayersWeights[a][i * prevCount + j] += this.learningRate * grad;
                }
                this.hiddenLayersBiases[a][i] += this.learningRate * deltasHidden[a][i];
            }
        }

        return true;
    }
}