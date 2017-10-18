"""TODO: document this."""
import genetic
import datetime
import random
import unittest
import string


class GuessPasswordTests(unittest.TestCase):
    """TODO: document this."""

    geneset = string.ascii_letters + " .!"

    def test_Hello_World(self):
        """TODO: document this."""
        target = "Hello World!"
        self.guess_password(target)
    #

    def test_Long_Password(self):
        """TODO: document this."""
        target = "King James. Bible!"
        self.guess_password(target)
    #

    def test_Random(self):
        """TODO: document this."""
        length = 150
        target = ''.join(random.choice(self.geneset) for _ in range(length))
        self.guess_password(target)
    #

    def test_benchmark(self):
        """TODO: document this."""
        genetic.Benchmark.run(self.test_Random)
    #

    def guess_password(self, target):
        """TODO: document this."""
        startTime = datetime.datetime.now()

        def fnGetFitness(genes):
            return get_fitness(genes, target)

        def fnDisplay(candidate):
            display(candidate, startTime)

        optimalFitness = len(target)
        best = genetic.get_best(fnGetFitness, len(target),
                                optimalFitness, self.geneset, fnDisplay)
        self.assertEqual(best.genes, target)
    #
#


def get_fitness(genes, target):
    """TODO: document this."""
    return sum(1 for expected, actual in zip(target, genes)
               if expected == actual)
#


def display(candidate, startTime):
    """TODO: document this."""
    timeDiff = datetime.datetime.now() - startTime
    print("{0}\t{1}\t{2!s}".format(candidate.genes,
                                   candidate.fitness, timeDiff))
#


if __name__ == "__main__":
    unittest.main()
