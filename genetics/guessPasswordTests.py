import genetic
import datetime
import random
import unittest
import string


class GuessPasswordTests(unittest.TestCase):
    #geneset = string.printable
    geneset = string.ascii_letters + " .!"
    
    def test_Hello_World(self):
        target = "Hello World!"
        self.guess_password(target)
    #
    
    def test_Long_Password(self):
        target = "King James. Bible!"
        self.guess_password(target)
    #
    
    def test_Random(self):
        length = 150
        target = ''.join(random.choice(self.geneset) for _ in range(length))
        self.guess_password(target)
    #

    def test_benchmark(self):
        genetic.Benchmark.run(self.test_Random)
    #
    
    def guess_password(self, target):
        startTime = datetime.datetime.now()
        
        def fnGetFitness(genes):
            return get_fitness(genes, target)
        
        def fnDisplay(candidate):
            display(candidate, startTime)
        
        optimalFitness = len(target)
        best = genetic.get_best(fnGetFitness, len(target), optimalFitness, self.geneset, fnDisplay)
        self.assertEqual(best.genes, target)
    #
#

def get_fitness(genes, target):
    return sum(1 for expected, actual in zip(target, genes) if expected == actual)
#

def display(candidate, startTime):
    timeDiff = datetime.datetime.now() - startTime
    print("{0}\t{1}\t{2!s}".format(candidate.genes, candidate.fitness, timeDiff))
#

if __name__ == "__main__":
    unittest.main()