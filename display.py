from matplotlib import pyplot as plt
from math import cos, sin, atan, degrees
import time


class Neuron():
    def __init__(self, x, y, index):
        self.x = x
        self.y = y
        self.index = index

    def draw(self, neuron_radius, active=False):
        circle = plt.Circle(
            (self.x, self.y), radius=neuron_radius, fill=active)
        plt.gca().add_patch(circle)


class Layer():
    def __init__(self, network, num_neurons, num_neurons_in_widest_layer,
                 weights):
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

    def _line_between_two_neurons(self, neuron1, neuron2, number=False):
        weight = self.weights[neuron1.index, neuron2.index]
        angle = atan((neuron2.x - neuron1.x) / float(neuron2.y - neuron1.y))
        x_adjustment = self.neuron_radius * sin(angle)
        y_adjustment = self.neuron_radius * cos(angle)
        xdata = (neuron1.x - x_adjustment, neuron2.x + x_adjustment)
        ydata = (neuron1.y - y_adjustment, neuron2.y + y_adjustment)
        line = plt.Line2D(xdata, ydata, linewidth=abs(weight) + 0.5)
        plt.gca().add_line(line)
        if not number:
            return
        textx = sum(xdata) / 2
        texty = sum(ydata) / 2
        if (angle < 0):
            plt.text(
                textx + x_adjustment * 5.5,
                texty + y_adjustment * 4,
                "{0:.2f}".format(weight),
                fontsize=9,
                family='monospace',
                weight='bold',
                rotation=270 - degrees(angle))
        else:
            plt.text(
                textx - x_adjustment * 5.5,
                texty - y_adjustment * 4,
                "{0:.2f}".format(weight),
                fontsize=9,
                family='monospace',
                weight='bold',
                rotation=90 - degrees(angle))

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
            plt.text(x_text, self.y, 'Input Layer', fontsize=11)
        elif layerType == -1:
            plt.text(x_text, self.y, 'Output Layer', fontsize=11)
        else:
            plt.text(
                x_text, self.y, 'Hidden Layer ' + str(layerType), fontsize=11)


class NeuralNetwork():
    def __init__(self, num_neurons_in_widest_layer):
        self.num_neurons_in_widest_layer = num_neurons_in_widest_layer
        self.layers = []
        self.layertype = 0

    def add_layer(self, num_neurons, weights):
        layer = Layer(self, num_neurons, self.num_neurons_in_widest_layer,
                      weights)
        self.layers.append(layer)

    def draw(self):
        for i in range(len(self.layers)):
            layer = self.layers[i]
            if i == len(self.layers) - 1:
                i = -1
            layer.draw(i)
        plt.axis('scaled')
        plt.axis('off')
        plt.title('Neural Network architecture', fontsize=15)


class DrawNN():
    def __init__(self, neural_network_shape, weights):
        self.neural_network_shape = neural_network_shape
        self.weights = weights

    def draw(self):
        widest_layer = max(self.neural_network_shape)
        network = NeuralNetwork(widest_layer)
        for i, item in enumerate(self.neural_network_shape):
            network.add_layer(item, self.weights[i - 1] if i != 0 else None)
        network.draw()


def displayNetwork(network_shape, network_weights):
    g = DrawNN(network_shape, network_weights)
    plt.figure(1)
    g.draw()
    plt.show(block=False)


class Visualizer:
    def __init__(self):

        self.best_fit = []
        self.avg_fit = []
        self.fitness_diversity = [0]
        self.gene_diversity = [0]
        self.hits = [0]
        self.misses = [0]
        self.mutes = [0]
        self.elites = [0]
        self.crossovers = [0]
        self.news = [0]
        self.max_fit = 0

        plt.ion()
        plt.figure(1)
        plt.subplot(411)
        # plot best and avg fitness
        self.best_fit_plt, = plt.plot(self.best_fit, label='Best Fitness')
        self.avg_fit_plt, = plt.plot(self.avg_fit, label='Average Fitness')
        self.best_fit_text_plt = plt.text(
            0.01,
            0.035,
            "Best Fitness: {0:.3f}".format(0),
            fontweight='bold',
            fontsize=9,
            transform=self.best_fit_plt.axes.transAxes)
        plt.xlim(xmin=0)
        plt.autoscale()
        plt.legend(loc=2, fontsize=6)
        plt.subplot(412)
        # plot fitness
        self.fitness_diversity_plt, = plt.plot(
            self.fitness_diversity,
            label='Fitness Diversity (Fitness Standard Deviation)')
        self.gene_diversity_plt, = plt.plot(
            self.gene_diversity,
            label='Gene Diversity (Mean Allele Standard Deviation)')
        plt.xlim(xmin=0)
        plt.autoscale()
        plt.legend(loc=2, fontsize=6)
        plt.subplot(413)
        # plot avg hits and misses of best so far, show max fitness so far
        self.hits_plt, = plt.plot(self.hits, label='Maximum Average Hits')
        self.misses_plt, = plt.plot(
            self.misses, label='Maximum Average Misses')
        self.max_fit_plt = plt.text(
            0.01,
            0.035,
            "Maximum Fitness: {0:.3f}".format(self.max_fit),
            fontweight='bold',
            fontsize=9,
            transform=self.hits_plt.axes.transAxes)
        plt.xlim(xmin=0)
        plt.autoscale()
        plt.legend(loc=2, fontsize=6)
        plt.subplot(414)

        self.mutes_plt, = plt.plot(self.mutes, label='Mutated Individuals')
        self.elites_plt, = plt.plot(self.elites, label='Elite Individuals')
        self.crossovers_plt, = plt.plot(
            self.crossovers, label='Crossover Individuals')
        self.news_plt, = plt.plot(self.news, label='New Individuals')

        plt.xlim(xmin=0)
        plt.ylim(ymin=0)
        plt.autoscale()
        plt.legend(loc=2, fontsize=6)
        plt.tight_layout()
        plt.gcf().canvas.draw()
        plt.show(block=False)

    def stop(self):
        plt.close(1)

    def refresh_data(self):
        self.best_fit_plt.set_data(range(len(self.best_fit)), self.best_fit)
        self.avg_fit_plt.set_data(range(len(self.avg_fit)), self.avg_fit)
        self.best_fit_text_plt.set_text("Best Fitness: {0:.3f}".format(
            self.best_fit[-1]))
        self.fitness_diversity_plt.set_data(
            range(len(self.fitness_diversity)), self.fitness_diversity)
        self.gene_diversity_plt.set_data(
            range(len(self.gene_diversity)), self.gene_diversity)
        self.hits_plt.set_data(range(len(self.hits)), self.hits)
        self.misses_plt.set_data(range(len(self.misses)), self.misses)
        self.max_fit_plt.set_text("Maximum Fitness: {0:.3f}".format(
            self.max_fit))
        self.mutes_plt.set_data(range(len(self.mutes)), self.mutes)
        self.elites_plt.set_data(range(len(self.elites)), self.elites)
        self.crossovers_plt.set_data(
            range(len(self.crossovers)), self.crossovers)
        self.news_plt.set_data(range(len(self.news)), self.news)

    def rescale(self):
        self.best_fit_plt.axes.relim()
        self.best_fit_plt.axes.autoscale_view(True, True, True)
        self.fitness_diversity_plt.axes.relim()
        self.fitness_diversity_plt.axes.autoscale_view(True, True, True)
        self.hits_plt.axes.relim()
        self.hits_plt.axes.autoscale_view(True, True, True)
        self.mutes_plt.axes.relim()
        self.mutes_plt.axes.autoscale_view(True, True, True)

    def repaint(self):
        self.rescale()
        # update canvas
        plt.gcf().canvas.draw()

    def add_generation(self, best_fit, avg_fit, diversity, best_hits,
                       best_misses, info):
        self.best_fit.append(best_fit)
        self.avg_fit.append(avg_fit)
        self.fitness_diversity.append(diversity[0])
        self.gene_diversity.append(diversity[1])
        self.mutes.append(info[0])
        self.elites.append(info[1])
        self.crossovers.append(info[2])
        self.news.append(info[3])

        if (best_fit > self.max_fit):
            self.hits.append(best_hits)
            self.misses.append(best_misses)
            self.max_fit = best_fit

        self.refresh_data()

        self.repaint()


if __name__ == "__main__":
    vis = Visualizer()
    x = 0.253
    y = 0
    z = 3
    while (True):
        start = time.perf_counter()
        x += 0.6132123
        y += 1
        z += 2
        vis.add_generation(x, x / 2.53, z * 1.64, x * 3, x * 2 / 5, 4)
        print("TIME " + str(y) + ": " + str(time.perf_counter() - start))
