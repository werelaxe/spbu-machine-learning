from time import time

from activation_functions import ActivationFunction
from dataset_ops import read_dataset
from layer import Layer
from neural_network import NeuralNetwork

TRAIN_DATASET_FILE_PATH = "dataset/mnist_train.csv"
TEST_DATASET_FILE_PATH = "dataset/mnist_test.csv"


def perform_two_layers_nn(train_input, train_output, test_input, test_output):
    nn = NeuralNetwork()
    nn.add_layer(Layer(shape=(784, 300), activation_function=ActivationFunction.SIGMOID))
    nn.add_layer(Layer(shape=(300, 10), activation_function=ActivationFunction.SOFTMAX))

    start = time()
    print('start pre-training')
    nn.learn(train_input[:100], train_output[:100], 0.01, 0.49, 100)

    print('start main training')
    nn.learn(train_input, train_output, 0.1, 0.75, 100)

    print("time:", time() - start)

    print("mse: ", nn.mse(test_input, test_output))
    print("accuracy: ", nn.accuracy(test_input, test_output))


def one_layer_nn(train_input, train_output, test_input, test_output):
    nn = NeuralNetwork([784, 10], activation_function=ActivationFunction.SOFTMAX)

    start = time()
    print('start pre-training')
    nn.learn(train_input, train_output, 0.01, 0.9, 100)
    print("time", time() - start)

    print("mse: ", nn.mse(test_input, test_output))
    print("accuracy: ", nn.accuracy(test_input, test_output))


def main():
    train_input, train_output = read_dataset(TRAIN_DATASET_FILE_PATH)
    test_input, test_output = read_dataset(TEST_DATASET_FILE_PATH)
    datasets = [train_input, train_output, test_input, test_output]

    one_layer_nn(*datasets)
    # time 1.6419239044189453
    # mse:  0.12406724153194079
    # accuracy:  0.9035

    perform_two_layers_nn(*datasets)


if __name__ == '__main__':
    main()
