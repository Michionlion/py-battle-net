import unittest
import datetime
import genetic
import random


class Fitness:

    def __init__(self, diffSum):
        self.diffSum = diffSum

    def __gt__(self, other):
        return self.diffSum < other.diffSum

    def __str__(self):
        return "{}".format(self.diffSum)


class MagicSquareTests(unittest.TestCase):

    def test_size_3(self):
        self.generate(3, 50)

    def test_size_5(self):
        self.generate(5, 500)

    def generate(self, diagSize, maxAge):
        nSqr = diagSize * diagSize
        geneset = [i for i in range(1, nSqr + 1)]
        expectedSum = diagSize * (nSqr + 1) / 2
        geneIndexes = [i for i in range(0, len(geneset))]

        def fnGetFitness(genes):
            return get_fitness(genes, diagSize, expectedSum)

        def fnCustomCreate():
            return random.sample(geneset, len(geneset))

        def fnDisplay(candidate):
            display(candidate, diagSize, startTime)

        def fnMutate(genes):
            mutate(genes, geneIndexes)

        optVal = Fitness(0)
        startTime = datetime.datetime.now()
        best = genetic.get_best(fnGetFitness, nSqr, optVal, geneset, fnDisplay,
                                fnMutate, fnCustomCreate, maxAge)

        self.assertTrue(not optVal > best.fitness)


def mutate(genes, indexes):
    indexA, indexB = random.sample(indexes, 2)
    genes[indexA], genes[indexB] = genes[indexB], genes[indexA]


def get_fitness(genes, diagSize, expectedSum):

    rows, cols, NEDSum, SEDSum = get_sums(genes, diagSize)

    diffs = sum(
        int(abs(s - expectedSum)) for s in rows + cols + [SEDSum, NEDSum]
        if s != expectedSum)

    return Fitness(diffs)


def get_sums(genes, diagSize):
    rows = [0 for _ in range(diagSize)]
    cols = [0 for _ in range(diagSize)]
    SEDSum = 0
    NEDSum = 0

    for row in range(diagSize):
        for col in range(diagSize):
            val = genes[row * diagSize + col]
            rows[row] += val
            cols[col] += val
        SEDSum += genes[row * diagSize + row]
        NEDSum += genes[row * diagSize + (diagSize - 1 - row)]

    return rows, cols, NEDSum, SEDSum


def display(candidate, diagSize, startTime):
    timeDiff = datetime.datetime.now() - startTime

    rows, cols, NEDSum, SEDSum = get_sums(candidate.genes, diagSize)

    for rowNum in range(diagSize):
        row = candidate.genes[rowNum * diagSize:(rowNum + 1) * diagSize]
        print("\t ", row, "=", rows[rowNum])
    print(NEDSum, "\t", cols, "\t", SEDSum)
    print(" - - - - - - - - - - -", candidate.fitness, timeDiff)

if __name__ == "__main__":
    unittest.main()
