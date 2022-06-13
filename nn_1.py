import numpy
import numpy as np
import scipy.special
import matplotlib.pyplot as plt


class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes,
                 learning_rate):
        self.i_nodes = input_nodes
        self.h_nodes = hidden_nodes
        self.o_nodes = output_nodes
        self.lr = learning_rate

        self.wih = np.random.normal(0.0, pow(self.h_nodes, -0.5), (self.h_nodes, self.i_nodes))
        self.who = np.random.normal(0.0, pow(self.o_nodes, -0.5), (self.o_nodes, self.h_nodes))

        self.activation_function = lambda x: scipy.special.expit(x)
        pass

    def train(self, inputs_list, targets_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        # print(final_outputs)

        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T, output_errors)

        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                     np.transpose(hidden_errors))
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                     np.transpose(inputs))

        pass

    def query(self, inputs_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs


input_nodes = 784
hidden_nodes = 100
output_nodes = 10

learning_rate = 0.1

training_data_file = open("data/mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

epochs = 5

w = 0
chart_x = np.arange(100, 10000, 50)
chart_y = []
# print(chart_x)
for charts in chart_x:
    n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
    for e in range(epochs):
        for record in training_data_list:
            # for w in range(100):
            all_values = record.split(',')
            # print(len(all_values))
            all_values[784] = '0'
            # print(all_values)
            for i in range(785):
                all_values[i] = int(all_values[i])
            # print(all_values[784])
            inputs = (np.asarray(all_values[1:]) / 255.0 * 0.99) + 0.01
            targets = np.zeros(output_nodes) + 0.01
            targets[int(all_values[0])] = 0.99
            n.train(inputs, targets)
            # print(w)
            w += 1
            # print(w)
            if w >= 2000:
                w = 0
                break
        # print(e)
        pass

    testing_data_file = open("data/mnist_test_100.csv", 'r')
    testing_data_list = testing_data_file.readlines()
    testing_data_file.close()

    items = 0
    item = 0

    for tests in testing_data_list:
        values = tests.split('\t')
        # print(type(tests[784]))
        # tests[784] = '0'
        for i in range(785):
            # print(tests[i])
            values[i] = int(values[i])
        inputs = (np.asarray(values[1:]) / 255.0 * 0.99) + 0.01
        out = n.query(inputs)
        result = out[values[0]]
        item += 1
        # print(type(result))
        if result >= 0.5:
            # print(True)
            items += 1
    percent = items / item
    chart_y.append(percent)
    print(percent)
    print(charts)

plt.plot(chart_x, chart_y)
plt.show()
