"""
Genetic Engine Module.

Provides methods and classes for evolving a phenotype with a given genotype.
"""
import random
from bisect import bisect_left
from math import exp


class Chromosome:
    """Holds gene and fitness values of one genotype-phenotype pair."""

    def __init__(self, genes, fitness):
        """Initialize Chromosome with given values."""
        self.genes = genes
        self.fitness = fitness
        self.age = 0


def _mutate(parent, geneSet, get_fitness):
    """Private method to mutate a set of genes and return a new Chromosome."""
    index = random.randrange(0, len(parent.genes))
    genes = parent.genes[:]
    newGene, alt = random.sample(geneSet, 2)
    genes[index] = alt if newGene == genes[index] else newGene
    fitness = get_fitness(genes)
    return Chromosome(genes, fitness)


def _mutate_custom(parent, custom_mutate, get_fitness):
    genes = parent.genes[:]
    custom_mutate(genes)
    fitness = get_fitness(genes)
    return Chromosome(genes, fitness)


def _generate_parent(length, geneSet, get_fitness):
    """Private method to generate a new parent from given information."""
    genes = []
    while len(genes) < length:
        sampleSize = min(length - len(genes), len(geneSet))
        genes.extend(random.sample(geneSet, sampleSize))
    fitness = get_fitness(genes)
    return Chromosome(genes, fitness)


def _get_improvement(new_child, generate_parent, maxAge):
    gen = 0
    parent = bestParent = generate_parent()
    yield bestParent, gen
    historicalFitnesses = [bestParent.fitness]
    while True:
        gen += 1
        # print("gen: {}".format(gen))
        child = new_child(parent)
        if parent.fitness > child.fitness:
            # bestParent is better, so keep it and keep going
            if maxAge is None:
                continue
            parent.age += 1
            if maxAge > parent.age:
                continue
            index = bisect_left(historicalFitnesses, child.fitness, 0,
                                len(historicalFitnesses))
            difference = len(historicalFitnesses) - index

            proportionSimilar = difference / len(historicalFitnesses)

            if random.random() < exp(-proportionSimilar):
                parent = child
                continue
            parent = bestParent
            parent.age = 0
            continue
        if not child.fitness > parent.fitness:
            # they are equal, so switch to the newer genetic line and continue
            child.age = parent.age + 1
            parent = child
            continue
        parent = child
        parent.age = 0
        if child.fitness > bestParent.fitness:
            yield child, gen
            bestParent = child
            historicalFitnesses.append(child.fitness)


def get_best(get_fitness,
             targetLen,
             optimalFitness,
             geneSet,
             display,
             custom_mutate=None,
             custom_create=None,
             maxAge=None):
    """Execute genetic algorithm with given information."""
    random.seed()

    if custom_mutate is None:

        def fnMutate(parent):
            return _mutate(parent, geneSet, get_fitness)
    else:

        def fnMutate(parent):
            return _mutate_custom(parent, custom_mutate, get_fitness)

    if custom_create is None:

        def fnGenerateParent():
            return _generate_parent(targetLen, geneSet, get_fitness)
    else:

        def fnGenerateParent():
            genes = custom_create()
            return Chromosome(genes, get_fitness(genes))

    for impv, gens in _get_improvement(fnMutate, fnGenerateParent, maxAge):
        display(impv)
        if not optimalFitness > impv.fitness:
            print("done in {0} generations".format(gens))
            return impv
