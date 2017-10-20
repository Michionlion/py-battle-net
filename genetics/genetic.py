"""
Genetic Engine Module.

Provides methods and classes for evolving a phenotype with a given genotype.
"""
import random
import time
import sys
import statistics as stats


class Chromosome:
    """Holds gene and fitness values of one genotype-phenotype pair."""

    def __init__(self, genes, fitness):
        """Initialize Chromosome with given values."""
        self.genes = genes
        self.fitness = fitness
#


class Benchmark:
    """Benchmark a given function using this class's static run method."""

    @staticmethod
    def run(function, output=False):
        """Benchmark function, optionally hiding or displaying output."""
        timings = []
        if not output:
            stdout = sys.stdout
        for i in range(100):
            if not output:
                sys.stdout = None
            startTime = time.time()
            function()
            seconds = time.time() - startTime
            timings.append(seconds)
            mean = stats.mean(timings)
            if not output:
                sys.stdout = stdout
            if i < 10 or i % 10 == 9:
                print("Run: {0}  Mean: {1:3.2f}  Stdev: {2:3.2f}"
                      .format(1 + i, mean, stats.stdev(timings, mean)
                              if i > 1 else 0))
#


def _mutate(parent, geneSet, get_fitness):
    """Private method to mutate a set of genes and return a new Chromosome."""
    index = random.randrange(0, len(parent.genes))
    genes = parent.genes[:]
    newGene, alt = random.sample(geneSet, 2)
    genes[index] = alt if newGene == genes[index] else newGene
    fitness = get_fitness(genes)
    return Chromosome(genes, fitness)
#


def _generate_parent(length, geneSet, get_fitness):
    """Private method to generate a new parent from geneSet of the given length."""
    genes = []
    while len(genes) < length:
        sampleSize = min(length - len(genes), len(geneSet))
        genes.extend(random.sample(geneSet, sampleSize))
    fitness = get_fitness(genes)
    return Chromosome(genes, fitness)
#


def _get_improvement(new_child, generate_parent):
    gen = 0
    bestParent = generate_parent()
    yield bestParent, gen
    while True:
        gen += 1
        child = new_child(bestParent)
        if bestParent.fitness > child.fitness:
            # bestParent is better, so keep it and keep going
            continue
        if not child.fitness > bestParent.fitness:
            # they are equal, so switch to the newer genetic line and keep going
            bestParent = child
            continue
        # child is better, so return child
        yield child, gen
        bestParent = child
#


def get_best(get_fitness, targetLen, optimalFitness, geneSet, display):
    """Execute genetic algorithm with given information."""
    random.seed()

    def fnMutate(parent):
        return _mutate(parent, geneSet, get_fitness)

    def fnGenerateParent():
        return _generate_parent(targetLen, geneSet, get_fitness)

    for improvement, generations in _get_improvement(fnMutate, fnGenerateParent):
        display(improvement)
        if not optimalFitness > improvement.fitness:
            print("done in {0} generations".format(generations))
            return improvement
#
