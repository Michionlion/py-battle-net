"""TODO: document this."""
import numpy as np
import neuralnet
import unittest
import argparse
import pickle
import random
import re
import datetime
from enum import Enum
from genetics import genetic
from multiprocessing import Pool


class Fitness:
    """Fitness object to compare genes"""

    def __init__(self, fails, repeats, tries, misses):
        self.fails = fails
        self.repeats = repeats
        self.tries = tries
        self.misses = misses

    def __gt__(self, other):
        if self.fails != other.fails:
            return self.fails < other.fails
        elif self.repeats != other.repeats:
            return self.repeats < other.repeats
        elif self.tries != other.tries:
            return self.tries < other.tries
        else:
            return self.misses < other.misses

    def __str__(self):
        return "{:.3f} fails, {:.3f} repeats, {:.3f} tries, {:.3f} misses".format(
            self.fails, self.repeats, self.tries, self.misses)


class BattleshipTests(unittest.TestCase):
    """TODO: document this."""

    # NETWORK_SHAPE = (8 * 8, 8 * 6, 8 * 6, 8 * 8)
    NETWORK_SHAPE = (8, 3, 8)
    # weights go from -10 to 10 (inclusive)
    WEIGHT_REACH = 10
    # each possible weight value differs by 0.001
    WEIGHT_DIFF = 100

    NUM_FITNESS_TESTS = 12

    GENE_LENGTH = 1
    for s in NETWORK_SHAPE:
        GENE_LENGTH *= s

    @unittest.skip("skipping random_network test")
    def test_random_network(self):

        nn = neuralnet.NeuralNetwork(self.NETWORK_SHAPE)
        genes = neuralnet.flatten(nn.weights)
        fit = self.get_fitness(genes)

        print("fitness: " + str(fit))

    @unittest.skip("skipping mutation test")
    def test_mutations(self):
        for _ in range(10):
            sample_genes = [-4.32, 1.5643, 0.2523, -12.432]
            print("genes: " + str(sample_genes))

            self.mutate_genes(sample_genes)

            print("after mutation: " + str(sample_genes))

    start = 0

    def test(self):
        """TODO: document this."""

        startTime = datetime.datetime.now()

        def fnDisplay(candidate):
            self.start += 1
            print(
                str(self.start) + " " + str(candidate.fitness) + " -- " +
                str(datetime.datetime.now() - startTime))

        def fnGetFitness(genes):
            return self.get_fitness(genes)

        def fnCreate():
            return self.create_gene()

        def fnMutate(genes):
            self.mutate_genes(genes)

        optimalFitness = Fitness(0, 0, 0, 0)
        best = genetic.get_best(fnGetFitness, None, optimalFitness, None,
                                fnDisplay, fnMutate, fnCreate, 500)
        print("Finished with best network: " + str(best.fitness) +
              " from genes: " + str(best.genes))

        unflat = neuralnet.unflatten(self.NETWORK_SHAPE, best.genes)
        nn = neuralnet.NeuralNetwork(self.NETWORK_SHAPE, weights=unflat)

        print(neuralnet.netinfo(nn))

        with open("weights.dat", 'wb') as file:
            pickle.dump(nn.weights, file, protocol=-1)

    def create_gene(self):
        """TODO: document this."""
        nn = neuralnet.NeuralNetwork(self.NETWORK_SHAPE)
        genes = neuralnet.flatten(nn.weights)
        return genes

    def mutate_genes(self, genes):
        # types of mutations
        def replace():
            # print("replace")
            index = random.randrange(0, len(genes))
            genes[index] = random.uniform(-self.WEIGHT_REACH,
                                          self.WEIGHT_REACH)

        def scale():
            # print("scale")
            index = random.randrange(0, len(genes))
            genes[index] *= random.uniform(0.5, 1.5)

        def delta_change():
            # print("delta")
            index = random.randrange(0, len(genes))
            change = random.uniform(-1, 1)
            genes[index] += change

        def sign_change():
            # print("sign")
            index = random.randrange(0, len(genes))
            genes[index] *= -1

        def swap():
            # print("swap")
            first = second = 0
            while (first == second):
                first = random.randrange(0, len(genes))
                second = random.randrange(0, len(genes))
            genes[first], genes[second] = genes[second], genes[first]

        mutations = [replace, scale, delta_change, sign_change, swap]
        numMutations = 1
        mutes = random.sample(mutations, numMutations)
        for mut in mutes:
            mut()

    def get_fitness(self, genes):
        """TODO: document this."""
        weights = neuralnet.unflatten(self.NETWORK_SHAPE, genes)
        network = neuralnet.NeuralNetwork(self.NETWORK_SHAPE, weights=weights)

        results = []
        pool = Pool()
        for i in range(self.NUM_FITNESS_TESTS):
            # running tests
            results.append(pool.apply_async(run_game, args=(network, )))
        # print(results)
        pool.close()
        pool.join()
        tries = fails = repeats = misses = 0
        for res in results:
            t, f, r, m = res.get()
            tries += t
            fails += f
            repeats += r
            misses += m

        return Fitness(fails / self.NUM_FITNESS_TESTS,
                       repeats / self.NUM_FITNESS_TESTS,
                       tries / self.NUM_FITNESS_TESTS / 8,
                       misses / self.NUM_FITNESS_TESTS)


def run_game(network):
    game = Game(size=(8, 1), ship_sizes=[4])
    # print("board:\n" + str(game.board))
    tries = fails = repeats = misses = 0
    while (not game.board.won()) and tries < game.board.squares() * 2.5:
        # input("continue?")
        inputVals = []
        for x in range(BattleshipTests.NETWORK_SHAPE[0]):
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
    return tries, fails, repeats, misses


class Ship:
    """TODO: document this."""

    DIR = Enum("DIR", "DOWN RIGHT")

    def __init__(self, pos, length):
        """Initialize a Ship object. If length < 0 the Ship is vertical."""
        self.length = abs(length)
        self.dir = self.DIR.DOWN if length < 0 else self.DIR.RIGHT
        self.pos = pos
        self.sectionsAlive = dict()

        for x in range(self.pos[0],
                       self.pos[0] + (self.length
                                      if self.dir is self.DIR.RIGHT else 1)):
            for y in range(self.pos[1], self.pos[1] +
                           (self.length if self.dir is self.DIR.DOWN else 1)):
                self.sectionsAlive[(x, y)] = True

        # print("length: " + str(self.length))
        # print("position: " + str(self.pos))
        # print("direction: " +
        # ("down" if self.dir is self.DIR.DOWN else "right"))
        # print("sections: " + str(self.sectionsAlive))

    def alive(self):
        """TODO: document this."""
        for alive in self.sectionsAlive.values():
            if alive:
                return True
        return False

    def hit(self, pos):
        """Return if the ship is hit by a shot at pos."""

        if pos in self.sectionsAlive:
            self.sectionsAlive[pos] = False
            # print("sectionsAlive: " + str(self.sectionsAlive))
            return True
        else:
            return False


class Board:
    """
    TODO: fully document this.

    shots is a dict with a tuple representing the position pointing to a value,
    1 for hit, -1 for miss. The key doesn't exist if it has not been shot at,
    but get_shot_at returns 0 for no-shot positions.
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

    def shoot(self, pos):
        """TODO: document this."""
        # print("shooting with x=" + str(x))
        if outween(pos[0], -1, self.size[0]) or outween(
                pos[1], -1, self.size[1]):
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

    def squares(self):
        """TODO: document this."""
        return self.size[0] * self.size[1]

    def get_shot_at(self, pos):
        """TODO: document this."""
        return self.shots[
            pos] if pos in self.shots else self.UNKNOWN_INT_MARKER

    def won(self):
        """TODO: document this."""
        if self.numshots() >= self.squares():
            return True
        for ship in self.ships:
            if ship.alive():
                return False
        return True

    def numshots(self):
        """TODO: document this."""
        return len(self.shots)

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


class Game:
    """TODO: document this."""

    def __init__(self, size=(5, 5), ship_sizes=[3, 2]):
        """TODO: FIXME for multiple ships - check ship overlap."""
        self._board_size = size
        self._ships = [self.genShip(length) for length in ship_sizes]
        self.board = Board(size, self._ships)

    def genShip(self, length):
        def rand(start, stop):
            if stop - start <= 1 or stop < start:
                return start
            else:
                return random.randrange(start, stop)

        d = random.choice([1, -1])

        xr = self._board_size[0] if d < 0 else self._board_size[0] - length
        yr = self._board_size[1] if d > 0 else self._board_size[1] - length

        shipPos = (rand(0, xr), rand(0, yr))

        return Ship(shipPos, d * length)


def human_play_test():
    """Playtest battleship with a human player (terminal input)."""

    size = tuple(map(int, input("Enter board size (w,h): ").split(",")))
    game = Game(size, [4])

    print("0, 0 is at the top left")
    while not game.board.won():
        print("board:\n" + str(game.board))

        done = False
        while not done:
            shot = input("Where do you want to shoot (x, y)? ")
            if (shot == "end"):
                break
            if re.match(r" *\d+ *, *\d+ *", shot) is None:
                print("could not parse input, use form 'number, number'!")
            else:
                done = True
        shot = tuple(map(int, shot.split(",")))
        print("Your shot was a " +
              ("hit!" if game.board.shoot(shot) else "miss..."))

    print("You Won! The final board was...")
    print(game.board)
    print(" - - WINNER!! - -")


def between(toTest, lower=0, upper=1):
    return toTest >= lower and toTest <= upper


def outween(toTest, lower=0, upper=1):
    return toTest <= lower or toTest >= upper


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="The classic game Battleship.")
    parser.add_argument(
        '--play',
        dest='test',
        action='store_const',
        const=True,
        default=False,
        help='Play the game as a human player (default: train AI)')
    args = parser.parse_args()
    if args.test:
        human_play_test()
    else:
        unittest.main()
