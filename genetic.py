"""
Genetic Engine Module.

Provides methods and classes for evolving a phenotype with a given genotype.
"""
import random

MAX_NUM_MUTATIONS = 5


class Chromosome:
    """Holds gene and fitness values of one genotype-phenotype pair."""

    def __init__(self, genes, fitness):
        """Initialize Chromosome with given values."""
        self.genes = genes
        self.fitness = fitness
        self.age = 0


def roulette_select(population):
    total = sum(ind.fitness for ind in population)

    def prob(choice):
        return choice.fitness / total

    r = random.random()
    total = 0
    for ind in population:
        total += prob(ind)
        if r < total:
            population.remove(
                ind
            )  # remove from population so that it doesn't get picked again
            return ind


def _mutate(genes, mutations, get_fitness):
    """Mutate genes with possible mutations mutations"""
    numMutations = random.randint(1, MAX_NUM_MUTATIONS)
    mutes = random.choices(mutations, k=numMutations)
    for mute in mutes:
        mute(genes)
    return genes


def chromify(population, get_fitness):
    for i in range(len(population)):
        population[i] = Chromosome(population[i], get_fitness(population[i]))


def evolve(pop_size, crossover_frac, mutation_rate, get_fitness, display,
           mutations, reproduce, create, max_age):
    gen = 0
    population = [create() for _ in range(pop_size)]
    chromify(population, get_fitness)

    while True:
        # create next gen
        # copy holdovers
        holdovers = [
            roulette_select(population)
            for _ in range((1 - crossover_frac) * pop_size)
        ]

        # crossover

        # mutate

        # evaluate

        gen += 1


if __name__ == "__main__":
    pop = [[3, 2], [2, 2, 6], [5, 8], [1, 5]]
    print(pop)
