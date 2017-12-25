"""
Genetic Engine Module.

Provides methods and classes for evolving a phenotype with a given genotype.
"""
import random

MAX_NUM_MUTATIONS = 1
N_POINT_CROSSOVER = True
N_POINT_CROSSOVER_RATE = 0.5
TOURNAMNET_SIZE_MODIFIER = 5
TOURNAMENT = True


class Chromosome:
    """Holds gene and fitness values of one genotype-phenotype pair."""

    def __init__(self, genes, fitness):
        """Initialize Chromosome with given values."""
        self.genes = genes
        self.fitness = fitness
        self.age = 0


def roulette_select(population):
    sum_total = sum(ind.fitness**1.25 for ind in population)

    # print("size of selecting " + str(len(population)) + " total " +
    #      str(sum_total))

    def prob(choice):
        return choice.fitness**1.25 / sum_total

    r = random.random()
    total = 0
    for ind in population:
        total += prob(ind)
        if r < total:
            # remove from population so that it doesn't get picked again
            # population.remove(ind)
            # print("selected individual with Fitness " + str(ind.fitness) +
            #      " with probability " + str(prob(ind)))
            return ind

    raise Exception(
        "Fell through roulette_select -- could be selecting on empty list!")


def tournament_select(population, size):

    if len(population) < size:
        size = len(population)

    pop = random.sample(population, size)
    # print("select")
    while len(pop) > 1:
        new_pop = []
        for n in range(1, len(pop), 2):
            if pop[n].fitness > pop[n - 1].fitness:
                new_pop.append(pop[n])
            else:
                new_pop.append(pop[n - 1])
        pop = new_pop
    # print("info: {0:.3f} {1:.3f}".format(len(pop), size) + ", selected " +
    #      str(pop[0].fitness))
    population.remove(pop[0])
    return pop[0]


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


def crossover(genes1, genes2):
    """Apply single or n point crossover to create children from given genes"""
    if N_POINT_CROSSOVER:
        n_point_crossover(genes1, genes2)
    else:
        single_point_crossover(genes1, genes2)


def n_point_crossover(genes1, genes2):
    """Do n-point crossover -- with given probability, exchange single alleles."""
    for i in range(len(genes1)):
        r = random.random()
        if r < N_POINT_CROSSOVER_RATE:
            genes1[i], genes2[i] = genes2[i], genes1[i]


def single_point_crossover(genes1, genes2):
    """Do single-point crossover to generate children from given genes"""
    r = random.randint(1, len(genes1) - 1)
    # print("crossing at " + str(r))

    genes1[r:], genes2[r:] = genes2[r:], genes1[r:]


def mutate(genes, mutations):
    """Mutate genes with possible mutations mutations"""
    if isinstance(genes, Chromosome):
        genes = genes.genes
    # num_mutations = random.randint(1, MAX_NUM_MUTATIONS)
    #mutes = random.choices(mutations, k=num_mutations)

    # for mute in mutes:
    #    genes = mute(genes)
    # print("applied " + str(mute))

    genes[:] = random.choice(mutations)(genes)

    return genes


def chromify(population, get_fitness):
    """TODO: document this."""

    # total_fit = 0

    def chrome(gene):
        # nonlocal total_fit
        if isinstance(gene, Chromosome):
            # total_fit += gene.fitness.score
            return gene
        else:
            fitness = get_fitness(gene)
            # total_fit += fitness.score
            return Chromosome(gene, fitness)

    for i, gene in enumerate(population):
        population[i] = chrome(gene)

    # for ind in population:
    #    ind.fitness.normalize(total_fit)


def dechromify(population):
    """TODO: document this."""
    for i, chrome in enumerate(population):
        population[i] = chrome.genes


def generate_next_population(population, reproduction_rates, mutation_rate,
                             mutations, create, select):
    elites = 0
    crossovers = 0
    news = 0
    next_pop = []
    i = len(population)
    while i > 0:
        r = random.random()
        if r < reproduction_rates[0]:
            # just copy over
            next_pop.append(select(population))
            next_pop.append(select(population))
            i -= 2
            elites += 2
        elif r < reproduction_rates[1]:
            # do crossover
            mom = select(population).genes
            dad = select(population).genes
            crossover(mom, dad)
            next_pop.append(mom)
            next_pop.append(dad)
            i -= 2
            crossovers += 2
        else:
            next_pop.append(create())
            next_pop.append(create())
            i -= 2
            news += 2

    # mutate
    num_mutes = 0
    for i, genes in enumerate(next_pop):
        r = random.random()
        if r < mutation_rate:
            next_pop[i] = mutate(genes, mutations)
            num_mutes += 1

    return next_pop, (num_mutes, elites, crossovers, news)


def evolve(pop_size, reproduction_rates, mutation_rate, get_fitness, display,
           mutations, create):
    """Execute a genetic algorithm with the given parameters - reproduction_rates = [relative_elite_rate, relative_crossover_rate, relative_new_rates]"""

    def select(population):
        if TOURNAMENT:
            return tournament_select(
                population,
                int(pop_size / TOURNAMNET_SIZE_MODIFIER) + 1)
        else:
            return roulette_select(population)

    total = sum(reproduction_rates)
    for i, rate in enumerate(reproduction_rates):
        reproduction_rates[i] = rate / total

    # rejigger repro rates for selection
    total = 0
    for i, rate in enumerate(reproduction_rates):
        reproduction_rates[i] += total
        total += rate

    gen = 0
    # ensure pop size is even
    pop_size = (pop_size // 2) * 2
    population = [create() for _ in range(pop_size)]
    chromify(population, get_fitness)

    gen_max = 10000
    while True:
        # create next gen
        population, gen_info = generate_next_population(
            population, reproduction_rates, mutation_rate, mutations, create,
            select)
        # evaluate
        chromify(population, get_fitness)
        gen += 1
        display(population, gen, gen_info)
        if (gen >= gen_max):
            cont = input('Continue? (y/n)')
            if cont.startswith('y'):
                gen_max += 1000
            else:
                break


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
