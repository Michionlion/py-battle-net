from matplotlib import pyplot
from math import cos, sin, atan


class Neuron():
    def __init__(self, x, y, index):
        self.x = x
        self.y = y
        self.index = index

    def draw(self, neuron_radius, active=False):
        circle = pyplot.Circle(
            (self.x, self.y), radius=neuron_radius, fill=active)
        pyplot.gca().add_patch(circle)


class Layer():
    def __init__(self, network, num_neurons, num_neurons_in_widest_layer, weights):
        self.vertical_layer_distance = 6
        self.horizontal_neuron_distance = 2
        self.neuron_radius = 0.5
        self.num_neurons_in_widest_layer = num_neurons_in_widest_layer
        self.weights = weights
        self.previous_layer = self._get_previous_layer(network)
        self.y = self._calculate_layer_y_position()
        self.neurons = self._init_neurons(num_neurons)

    def _init_neurons(self, num_neurons):
        neurons = []
        index = 0
        x = self._calculate_left_margin_so_layer_is_centered(num_neurons)
        for iteration in range(num_neurons):
            neuron = Neuron(x, self.y, index)
            neurons.append(neuron)
            x += self.horizontal_neuron_distance
            index += 1
        return neurons

    def _calculate_left_margin_so_layer_is_centered(self, num_neurons):
        return self.horizontal_neuron_distance * (
            self.num_neurons_in_widest_layer - num_neurons) / 2.0

    def _calculate_layer_y_position(self):
        if self.previous_layer:
            return self.previous_layer.y + self.vertical_layer_distance
        else:
            return 0

    def _get_previous_layer(self, network):
        if len(network.layers) > 0:
            return network.layers[-1]
        else:
            return None

    def _line_between_two_neurons(self, neuron1, neuron2):
        weight = self.weights[neuron1.index, neuron2.index]
        angle = atan((neuron2.x - neuron1.x) / float(neuron2.y - neuron1.y))
        x_adjustment = self.neuron_radius * sin(angle)
        y_adjustment = self.neuron_radius * cos(angle)
        xdata = (neuron1.x - x_adjustment, neuron2.x + x_adjustment)
        ydata = (neuron1.y - y_adjustment, neuron2.y + y_adjustment)
        line = pyplot.Line2D(xdata, ydata, linewidth=abs(weight))
        pyplot.gca().add_line(line)
        # pyplot.text(xdata[0] + xdata[1] / 2, ydata[0] + ydata[1] / 2, str(weight))

    def draw(self, layerType=0):
        for neuron in self.neurons:
            neuron.draw(self.neuron_radius)
            if self.previous_layer:
                for previous_layer_neuron in self.previous_layer.neurons:
                    self._line_between_two_neurons(neuron,
                                                    previous_layer_neuron)
        # write Text
        x_text = self.num_neurons_in_widest_layer * self.horizontal_neuron_distance
        if layerType == 0:
            pyplot.text(x_text, self.y, 'Input Layer', fontsize=12)
        elif layerType == -1:
            pyplot.text(x_text, self.y, 'Output Layer', fontsize=12)
        else:
            pyplot.text(
                x_text, self.y, 'Hidden Layer ' + str(layerType), fontsize=12)


class NeuralNetwork():
    def __init__(self, num_neurons_in_widest_layer):
        self.num_neurons_in_widest_layer = num_neurons_in_widest_layer
        self.layers = []
        self.layertype = 0

    def add_layer(self, num_neurons, weights):
        layer = Layer(self, num_neurons,
                      self.num_neurons_in_widest_layer, weights)
        self.layers.append(layer)

    def draw(self):
        pyplot.figure()
        for i in range(len(self.layers)):
            layer = self.layers[i]
            if i == len(self.layers) - 1:
                i = -1
            layer.draw(i)
        pyplot.axis('scaled')
        pyplot.axis('off')
        pyplot.title('Neural Network architecture', fontsize=15)
        pyplot.show(block=False)


class DrawNN():
    def __init__(self, neural_network_shape, weights):
        self.neural_network_shape = neural_network_shape
        self.weights = weights

    def draw(self):
        widest_layer = max(self.neural_network_shape)
        network = NeuralNetwork(widest_layer)
        for i, item in enumerate(self.neural_network_shape):
            network.add_layer(item, self.weights[i-1] if i != 0 else None)
        network.draw()

def displayNetwork(network_shape, network_weights):
    g = DrawNN(network_shape, network_weights)

    g.draw()
