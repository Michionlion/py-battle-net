"""
Genetic Engine Module.

Provides methods and classes for evolving a phenotype with a given genotype.
"""
import random
import time
import sys
import statistics as stats


class Chromosome:
    """TODO: document this."""

    def __init__(self, genes, fitness):
        """TODO: document this."""
        self.genes = genes
        self.fitness = fitness
#


class Benchmark:
    """TODO: document this."""

    @staticmethod
    def run(function, output=False):
        """TODO: document this."""
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
    """TODO: document this."""
    index = random.randrange(0, len(parent.genes))
    childGenes = list(parent.genes)
    newGene, alt = random.sample(geneSet, 2)
    childGenes[index] = alt if newGene == childGenes[index] else newGene

    genes = ''.join(childGenes)
    fitness = get_fitness(genes)
    return Chromosome(genes, fitness)
#


def _generate_parent(length, geneSet, get_fitness):
    genes = []
    while len(genes) < length:
        sampleSize = min(length - len(genes), len(geneSet))
        genes.extend(random.sample(geneSet, sampleSize))

    genes = ''.join(genes)
    fitness = get_fitness(genes)
    return Chromosome(genes, fitness)
#


def get_best(get_fitness, targetLen, optimalFitness, geneSet, display):
    """Execute genetic algorithm with given information."""
    gen = 0
    random.seed()
    bestParent = _generate_parent(targetLen, geneSet, get_fitness)
    display(bestParent)

    if bestParent.fitness >= optimalFitness:
        print("done in {0} generations".format(gen))
        return bestParent

    while True:
        gen += 1
        child = _mutate(bestParent, geneSet, get_fitness)

        if bestParent.fitness >= child.fitness:
            continue

        display(child)
        if child.fitness >= optimalFitness:
            print("done in {0} generations".format(gen))
            return child

        bestParent = child
#
