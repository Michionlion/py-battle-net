"""TODO: document this."""
import unittest
import datetime
import genetic


class EightQueensTests(unittest.TestCase):
    """TODO: document this."""

    @unittest.skip("skipping benchmark test")
    def test_benchmark(self):
        """TODO: document this."""
        genetic.Benchmark.run(lambda: self.test(20))

    def test(self, size=8):
        """TODO: document this."""
        geneset = [i for i in range(size)]
        startTime = datetime.datetime.now()

        def fnDisplay(candidate):
            display(candidate, startTime, size)

        def fnGetFitness(genes):
            return get_fitness(genes, size)

        optimalFitness = Fitness(0)
        best = genetic.get_best(fnGetFitness, 2 * size,
                                optimalFitness, geneset, fnDisplay)
        self.assertTrue(not optimalFitness > best.fitness)
#


class Board:
    """TODO: document this."""

    def __init__(self, genes, size):
        """TODO: document this."""
        board = [['.'] * size for _ in range(size)]
        for index in range(0, len(genes), 2):
            row = genes[index]
            column = genes[index + 1]
            board[column][row] = 'Q'
        self._board = board

    def get(self, row, column):
        """TODO: document this."""
        return self._board[column][row]

    def __str__(self):
        """TODO: document this."""
        """
        Prints the board.

        0,0 prints in the top right corner
        """
        str = ""
        for i in reversed(range(0, len(self._board))):
            str += ' '.join(self._board[i]) + '\n'

        return str
#


class Fitness:
    """TODO: document this."""

    def __init__(self, total):
        """TODO: document this."""
        self.total = total

    def __gt__(self, other):
        """TODO: document this."""
        return self.total < other.total

    def __str__(self):
        """TODO: document this."""
        return "{}".format(self.total)
#


def get_fitness(genes, size):
    """TODO: document this."""
    board = Board(genes, size)
    rowsWithQueens = set()
    colsWithQueens = set()
    northEastDiagonalsWithQueens = set()
    southEastDiagonalsWithQueens = set()
    for row in range(size):
        for col in range(size):
            if board.get(row, col) == 'Q':
                rowsWithQueens.add(row)
                colsWithQueens.add(col)
                northEastDiagonalsWithQueens.add(row + col)
                southEastDiagonalsWithQueens.add(size - 1 - row + col)

    total = size - len(rowsWithQueens) + size - \
        len(colsWithQueens) + size - \
        len(northEastDiagonalsWithQueens) + size - \
        len(southEastDiagonalsWithQueens)

    return Fitness(total)
#


def display(candidate, startTime, size):
    """TODO: document this."""
    timeDiff = datetime.datetime.now() - startTime
    board = Board(candidate.genes, size)
    print(str(board))
    print("{}\t- {}\t{}".format(' '.join(map(str, candidate.genes)),
                                candidate.fitness, timeDiff))


#
if __name__ == "__main__":
    unittest.main()
