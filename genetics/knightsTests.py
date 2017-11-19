"""TODO: document this."""
import random
import datetime
import unittest
import genetic


class Position:
    """TODO: document this."""

    def __init__(self, x, y):
        """TODO: document this."""
        self.x = x
        self.y = y

    def __str__(self):
        return "{},{}".format(self.x, self.y)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return self.x * 1000 + self.y


class KnightsTests(unittest.TestCase):
    """TODO: document this."""

    def test_10x10(self):
        self.find_knight_positions(10, 10, 22)

    @unittest.skip("skipping")
    def test_8x8(self):
        width = 8
        height = 8
        self.find_knight_positions(width, height, 14)

    @unittest.skip("skipping")
    def test_3x4(self):
        width = 4
        height = 3

        self.find_knight_positions(width, height, 6)

    @staticmethod
    def find_knight_positions(width, height, expectedKnights):
        startTime = datetime.datetime.now()

        def fnDisplay(candidate):
            display(candidate, startTime, width, height)

        def fnGetFitness(genes):
            return get_fitness(genes, width, height)

        def fnGetRandomPosition():
            return Position(random.randrange(0, width), random.randrange(0, height))

        def fnMutate(genes):
            mutate(genes, fnGetRandomPosition)

        def fnCreate():
            return create(fnGetRandomPosition, expectedKnights)

        optimalFitness = width * height
        genetic.get_best(
            fnGetFitness, None, optimalFitness, None, fnDisplay, fnMutate, fnCreate)
        # self.assertTrue(not optimalFitness > best.fitness)


class Board:
    def __init__(self, positions, width, height):
        board = [['.'] * width for _ in range(height)]

        for index in range(len(positions)):
            knightPosition = positions[index]
            board[knightPosition.y][knightPosition.x] = 'N'

        self._board = board
        self._width = width
        self._height = height

    def __str__(self):
        """TODO: document this."""
        """
        Prints the board.

        0,0 prints in the top right corner
        """
        for i in reversed(range(self._height)):
            print(i, "\t", ' '.join(self._board[i]))
        print(" \t", ' '.join(map(str, range(self._width))))
        return " "


def get_attacks(location, boardWidth, boardHeight):
    """TODO: document this."""
    return [i for i in set(
        Position(x + location.x, y + location.y)
        for x in [-2, -1, 1, 2] if 0 <= x + location.x < boardWidth
        for y in [-2, -1, 1, 2] if 0 <= y + location.y < boardHeight
        and abs(y) != abs(x)
    )]


def get_fitness(genes, width, height):
    attacked = set(
        pos for kn in genes for pos in get_attacks(kn, width, height))
    return len(attacked)


def create(fnGetRandomPosition, expectedKnights):
    """TODO: document this."""
    genes = [fnGetRandomPosition() for _ in range(expectedKnights)]
    return genes


def display(candidate, startTime, boardWidth, boardHeight):
    timeDiff = datetime.datetime.now() - startTime
    board = Board(candidate.genes, boardWidth, boardHeight)
    print(board)

    print("{}\n\t{}\t{}".format(
        ' '.join(map(str, candidate.genes)), candidate.fitness, timeDiff))


def mutate(genes, fnGetRandomPosition):
    index = random.randrange(0, len(genes))
    genes[index] = fnGetRandomPosition()


if __name__ == "__main__":
    unittest.main()
