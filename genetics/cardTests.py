"""TODO: document this."""
import unittest
import datetime
import genetic
import operator
import functools
import random


class CardTests(unittest.TestCase):
    """TODO: document this."""

    def test_benchmark(self):
        """TODO: document this."""
        genetic.Benchmark.run(lambda: self.test())

    def test(self):
        """TODO: document this."""
        geneset = [i + 1 for i in range(10)]
        startTime = datetime.datetime.now()

        def fnDisplay(candidate):
            display(candidate, startTime)

        def fnGetFitness(genes):
            return get_fitness(genes)

        def fnMutate(genes):
            mutate(genes, geneset)

        optimalFitness = Fitness(36, 360, 0)
        best = genetic.get_best(
            fnGetFitness, 10, optimalFitness,
            geneset, fnDisplay, custom_mutate=fnMutate)
        self.assertTrue(not optimalFitness > best.fitness)


class Fitness:
    """TODO: document this."""

    def __init__(self, group1Sum, group2Prod, dups):
        """TODO: document this."""
        self.group1Sum = group1Sum
        self.group2Prod = group2Prod
        self.diff = abs(36 - group1Sum) + abs(360 - group2Prod)
        self.dups = dups

    def __gt__(self, other):
        """TODO: document this."""
        if self.dups != other.dups:
            return self.dups < other.dups
        return self.diff < other.diff

    def __str__(self):
        """TODO: document this."""
        return "sum: {} prod: {} dups: {}".format(
            self.group1Sum, self.group2Prod, self.dups)
#


def mutate(genes, geneset):
    """TODO: document this."""
    if len(genes) == len(set(genes)):
        count = random.randint(1, 4)
        while count > 0:
            count -= 1
            indexA, indexB = random.sample(range(len(genes)), 2)
            genes[indexA], genes[indexB] = genes[indexB], genes[indexA]
    else:
        indexA = random.randrange(0, len(genes))
        indexB = random.randrange(0, len(geneset))
        genes[indexA] = geneset[indexB]
#


def get_fitness(genes):
    """TODO: document this."""
    group1Sum = sum(genes[0:5])
    group2Prod = functools.reduce(operator.mul, genes[5:10])
    dups = (len(genes) - len(set(genes)))
    return Fitness(group1Sum, group2Prod, dups)
#


def display(candidate, startTime):
    """TODO: document this."""
    timeDiff = datetime.datetime.now() - startTime
    print("{} - {}\t{}\t{}".format(
        ', '.join(map(str, candidate.genes[0:5])),
        ', '.join(map(str, candidate.genes[5:10])),
        candidate.fitness, timeDiff))
#


if __name__ == '__main__':
    unittest.main()
