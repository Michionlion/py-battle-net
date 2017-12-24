"""
Genetic Engine Module.

Provides methods and classes for evolving a phenotype with a given genotype.
"""
import random
from profilestats import profile

MAX_NUM_MUTATIONS = 7


class Chromosome:
    """Holds gene and fitness values of one genotype-phenotype pair."""

    def __init__(self, genes, fitness):
        """Initialize Chromosome with given values."""
        self.genes = genes
        self.fitness = fitness
        self.age = 0


def roulette_select(population):
    sum_total = sum(ind.fitness for ind in population)

    def prob(choice):
        return choice.fitness / sum_total

    r = random.random()
    total = 0
    for ind in population:
        total += prob(ind)
        if r < total:
            # remove from population so that it doesn't get picked again
            population.remove(ind)
            return ind

    raise Exception(
        "Fell through roulette_select -- could be selecting on empty list!")


"""
Options for custom crossover algorithms instead of basic single point
crossover as follows:

Swap either single weights or all weights for a given neuron in the network.
So for example, given two parents selected for reproduction either choose a
particular weight in the network and swap the value (for our swaps we produced
two offspring and then chose the one with the best fitness to survive in the
next generation of the population), or choose a particular neuron in the
network and swap all the weights for that neuron to produce two offspring.

Swap an entire layer's weights. So given parents A and B, choose a particular
layer (the same layer in both) and swap all the weights between them to
produce two offsping. This is a large move so we set it up so that this
operation would be selected less often than the others. Also, this may
not make sense if your network only has a few layers.
"""


@profile()
def crossover(genes1, genes2):
    """Do single-point crossover to generate children from given genes"""
    r = random.randint(1, len(genes1) - 1)
    # print("crossing at " + str(r))

    genes1[r:], genes2[r:] = genes2[r:], genes1[r:]


@profile()
def _mutate(genes, mutations):
    """Mutate genes with possible mutations mutations"""
    if isinstance(genes, Chromosome):
        genes = genes.genes
    num_mutations = random.randint(1, MAX_NUM_MUTATIONS)
    mutes = random.choices(mutations, k=num_mutations)

    for mute in mutes:
        genes = mute(genes)
    return genes, num_mutations


@profile()
def chromify(population, get_fitness):
    """TODO: document this."""

    def chrome(gene):
        if isinstance(gene, Chromosome):
            return gene
        else:
            return Chromosome(gene, get_fitness(gene))

    for i, gene in enumerate(population):
        population[i] = chrome(gene)


@profile()
def dechromify(population):
    """TODO: document this."""
    for i, chrome in enumerate(population):
        population[i] = chrome.genes


@profile()
def evolve(pop_size, crossover_rate, mutation_rate, get_fitness, display,
           mutations, create):
    """Execute a genetic algorithm with the given parameters"""
    gen = 0
    population = [create() for _ in range(pop_size)]
    chromify(population, get_fitness)

    while gen < 10:
        # create next gen
        nextPop = []
        i = pop_size
        while i > 0:
            r = random.random()
            if r > crossover_rate or len(population) < 2:
                # just copy over
                nextPop.append(roulette_select(population))
                i -= 1
            else:
                # do crossover
                mom = roulette_select(population).genes
                dad = roulette_select(population).genes
                crossover(mom, dad)
                nextPop.append(mom)
                nextPop.append(dad)
                i -= 2

        # mutate
        population = nextPop
        num_mutes = 0
        for i, genes in enumerate(population):
            r = random.random()
            if r < mutation_rate:
                population[i], mutes = _mutate(genes, mutations)
                num_mutes += mutes

        # evaluate
        chromify(population, get_fitness)
        gen += 1
        display(population, gen, num_mutes)


if __name__ == "__main__":
    # pop = [[3, 2], [2, 2, 6], [5, 8], [1, 5]]
    #
    # def fit(genes):
    #     return sum(genes)
    #
    # print(pop)
    # chromify(pop, fit)
    # print(pop)
    # dechromify(pop)
    # print(pop)
    gene1 = [6, 5, 4, 3, 2, 1, 0]
    gene2 = [7, 8, 9, 10, 11, 12, 13]
    crossover(gene1, gene2)
    print(gene1)
    print(gene2)
