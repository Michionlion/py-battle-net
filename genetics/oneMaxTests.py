import unittest
import datetime
import genetic


class OneMaxTests(unittest.TestCase):

    @unittest.skip("skipping benchmark test")
    def test_benchmark(self):
        genetic.Benchmark.run(lambda: self.test(4000))

    def test(self, length=100):
        geneset = [0, 1]
        startTime = datetime.datetime.now()

        def fnDisplay(candidate):
            display(candidate, startTime)

        def fnGetFitness(genes):
            return get_fitness(genes)

        optimalFitness = length
        best = genetic.get_best(fnGetFitness, length,
                                optimalFitness, geneset, fnDisplay)
        self.assertEqual(best.fitness, optimalFitness)
#


def get_fitness(genes):
    return genes.count(1)
#


def display(candidate, startTime):
    timeDiff = datetime.datetime.now() - startTime
    print("{}...{}\t{:3.2f}\t{}".format(
        ''.join(map(str, candidate.genes[:15])),
        ''.join(map(str, candidate.genes[-15:])),
        candidate.fitness, timeDiff))
#


if __name__ == "__main__":
    unittest.main()
