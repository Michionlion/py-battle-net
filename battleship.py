"""TODO: document this."""
import numpy as np
import neuralnet
import unittest
import argparse
import pickle
from genetics import genetic


class Fitness:
    """Fitness object to compare genes"""

    def __init__(self, fails, repeats, misses):
        self.fails = fails
        self.repeats = repeats
        self.misses = misses
    #

    def __gt__(self, other):
        if self.fails != other.fails:
            return self.fails < other.fails
        elif self.repeats != other.repeats:
            return self.repeats < other.repeats
        else:
            return self.misses < other.misses
    #

    def __str__(self):
        return "{0:.3f} invalid moves, {1:.3f} repeat moves, {2:.3f} misses per game".format(self.fails, self.repeats, self.misses)
#


class BattleshipTests(unittest.TestCase):
    """TODO: document this."""

    NETWORK_SHAPE = (8, 3, 8)
    # weights go from -10 to 10 (inclusive)
    WEIGHT_REACH = 10
    # each possible weight value differs by 0.001
    WEIGHT_DIFF = 100

    NUM_FITNESS_TESTS = 7

    @unittest.skip("skipping random_network test")
    def test_random_network(self):
        length = 1
        for s in self.NETWORK_SHAPE:
            length *= s

        rands = (self.WEIGHT_REACH * 2) * \
            np.random.sample(length) - self.WEIGHT_REACH
        print(rands)

        nn = neuralnet.NeuralNetwork(self.NETWORK_SHAPE)
        genes = neuralnet.flatten(nn.weights)
        fit = self.get_fitness(genes)

        print("fitness: " + str(fit))
    #

    def test(self):
        """TODO: document this."""
        geneset = [
            i / (self.WEIGHT_REACH * self.WEIGHT_DIFF)
            for i in range(-self.WEIGHT_DIFF *
                           self.WEIGHT_REACH**2,
                           self.WEIGHT_DIFF *
                           self.WEIGHT_REACH**2 + 1)]
        # print(geneset)

        length = 1
        for s in self.NETWORK_SHAPE:
            length *= s

        def fnDisplay(candidate):
            print(candidate.fitness)

        def fnGetFitness(genes):
            return self.get_fitness(genes)

        optimalFitness = Fitness(0, 0, 0)
        best = genetic.get_best(fnGetFitness, length,
                                optimalFitness, geneset, fnDisplay)
        print("Finished with best network: " +
              str(best.fitness) + " from genes: " + str(best.genes))

        unflat = neuralnet.unflatten(self.NETWORK_SHAPE, best.genes)
        nn = neuralnet.NeuralNetwork(self.NETWORK_SHAPE, weights=unflat)

        print(neuralnet.netinfo(nn))

        with open("weights.dat", 'wb') as file:
            pickle.dump(nn.weights, file, protocol=-1)

    def get_fitness(self, genes):
        """TODO: document this."""
        weights = neuralnet.unflatten(self.NETWORK_SHAPE, genes)
        network = neuralnet.NeuralNetwork(self.NETWORK_SHAPE, weights=weights)
        fails = 0
        misses = 0
        repeats = 0
        for i in range(self.NUM_FITNESS_TESTS):
            game = Game()
            tries = 0
            # allows tries to be more than normally possible so as to compare networks that are bad at not shooting at the same spot
            while (not game.board.won()) and tries < game.board.squares() * 2.5:
                # input("continue?")
                inputVals = []
                for x in range(self.NETWORK_SHAPE[0]):
                    inputVals.append(game.board.get_shot_at((x, 0)))
                #
                # print("evaluating on " + str(inputVals))
                selection = network.evaluate(inputVals)
                # print("outputs were " + str(selection))
                shot = (int(np.argmax(selection)), 0)
                # print("shooting at " + str(shot))
                result = game.board.shoot(shot)
                if result == -2:
                    fails += 1
                elif result == -3:
                    repeats += 1
                elif result is False:
                    misses += 1
                # print("board: " + str(game.board))
                tries += 1
        #
        return Fitness(fails / self.NUM_FITNESS_TESTS, repeats / self.NUM_FITNESS_TESTS, misses / self.NUM_FITNESS_TESTS)
#


class Ship:
    """TODO: document this."""

    def __init__(self, x, length):
        """TODO: FIXME for 2D."""
        self.length = int(length)
        self.x = int(x)
        self.sectionsAlive = [True] * (length)

        # print("length: " + str(self.length))
        # print("position: " + str(x))
        # print("sections: " + str(self.sectionsAlive))

    #
    def alive(self):
        """TODO: document this."""
        count = self.length
        for alive in self.sectionsAlive:
            if not alive:
                count -= 1
        #
        return count > 0
    #

    def hit(self, pos):
        """TODO: FIXME for 2D."""
        x = int(pos[0])
        # print("trying hit at " + str(x))
        if self.x <= x and self.x + self.length > x:
            self.sectionsAlive[x - self.x] = False
            # print("good hit")
            return True
        else:
            # print("bad hit")
            return False
    #

# Shipwidth


class Board:
    """
    TODO: fully document this.

    shots is a dict encoded with a tuple representing the position pointing to a value, 1 for hit, -1 for miss. The key doesn't exist if it has not been shot at, but get_shot_at returns 0 for no-shot positions.
    """

    HIT_INT_MARKER = 1
    HIT_STR_MARKER = "X"
    MISS_INT_MARKER = -1
    MISS_STR_MARKER = "O"
    UNKNOWN_INT_MARKER = 0
    UNKNOWN_STR_MARKER = "."

    def __init__(self, size, ships=[]):
        """TODO: document this."""
        self.ships = ships
        self.shots = dict()
        self.size = size
    #

    def shoot(self, pos):
        """TODO: document this."""
        # print("shooting with x=" + str(x))
        if pos[0] < 0 or pos[0] >= self.size[0] or pos[1] < 0 or pos[1] >= self.size[1]:
            # print("invalid position " + str(pos))
            return -2
        elif pos in self.shots:
            # print("already shot at " + str(pos))
            return -3
        else:
            for ship in self.ships:
                # print("hitting " + str(ship))
                if ship.hit(pos):
                    self.shots[pos] = self.HIT_INT_MARKER
                    return True

            self.shots[pos] = self.MISS_INT_MARKER
            return False
    #

    def squares(self):
        """TODO: document this."""
        return self.size[0] * self.size[1]

    def get_shot_at(self, pos):
        """TODO: document this."""
        return self.shots[pos] if pos in self.shots else self.UNKNOWN_INT_MARKER

    def won(self):
        """TODO: document this."""
        if self.numshots() >= self.squares():
            return True
        for ship in self.ships:
            if ship.alive():
                return False
        return True
    #

    def numshots(self):
        """TODO: document this."""
        return len(self.shots)
    #

    def __str__(self):
        """TODO: document this."""
        rep = ""
        for y in range(self.size[1]):
            for x in range(self.size[0]):
                shot = self.get_shot_at((x, y))
                if shot == self.UNKNOWN_INT_MARKER:
                    rep += self.UNKNOWN_STR_MARKER
                elif shot == self.HIT_INT_MARKER:
                    rep += self.HIT_STR_MARKER
                elif shot == self.MISS_INT_MARKER:
                    rep += self.MISS_STR_MARKER
                else:
                    rep += "E"
                if x < self.size[0] - 1:
                    rep += " "
            if y < self.size[1] - 1:
                rep += "\n"
        return rep
# Board


class Game:
    """TODO: document this."""

    SHIP_SIZES = [3]  # , 2, 1]

    def __init__(self, size=(8, 1)):
        """TODO: document this."""
        self._board_size = size
        self._ships = [Ship(np.random.randint(
            0, size[0] - length + 1), length) for length in self.SHIP_SIZES]
        self.board = Board(size, self._ships)

#


def human_play_test():
    """Playtest battleship with a human player (input entered through terminal)."""
    # NEED TO FIX THIS TO WORK WITH TUPLES FOR POSITIONS

    BOARD_SIZE = (10, 1)
    SHIP_SIZES = [3]  # , 2, 1]
    # NUM_SHIPS = 1

    ships = [Ship(np.random.randint(0, BOARD_SIZE[0] - length + 1), length)
             for length in SHIP_SIZES]
    board = Board(BOARD_SIZE, ships)

    while not board.won():
        shot = input("Where do you want to shoot? ")
        if(shot == "end"):
            break
        print("Your shot was a " +
              ("hit!" if board.shoot(int(shot)) else "miss..."))
        print("board:\n" + str(board))
#


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="The classic game Battleship.")
    parser.add_argument('--play', dest='test', action='store_const', const=True, default=False,
                        help='Play the game as a human player (default: train AI to play the game)')
    args = parser.parse_args()
    if args.test:
        human_play_test()
    else:
        unittest.main()
