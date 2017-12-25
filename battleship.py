"""TODO: document this."""
import numpy as np
import neuralnet
import genetic
import unittest
import argparse
import random
import re
import datetime
from enum import Enum
from display import Visualizer as Vis

# game, network, and fitness params
USE_INVALID = True
NEURAL_NET_SHAPE = (25, 25, 25)
SHIP_SIZE_LIST = [4, 3]
GAME_SIZE = (5, 5)
RANDOM_NETWORK_HIT_CHANCE = float(sum(SHIP_SIZE_LIST) / (GAME_SIZE[0]*GAME_SIZE[1]))


class Fitness:
    """Fitness object to compare genes"""

    def __init__(self,
                 fails,
                 repeats,
                 tries,
                 misses,
                 hits,
                 hit_log=None,
                 score=None):
        self.fails = fails
        self.repeats = repeats
        self.tries = tries
        self.misses = misses
        self.hits = hits
        if hit_log is None:
            self.score = 'nan' if score is None else score
        else:
            self.score = self.calculate_score(hit_log)

    def calculate_score(self, hit_log):
        total_reward = 0
        for t in range(len(hit_log)):
            total_reward += self.reward_at(t, hit_log)
        total_reward += 1 / len(hit_log)
        return total_reward if total_reward > 0 else 0

    def reward_at(self, t0, hit_log):
        reward = 0
        random_network_hit = RANDOM_NETWORK_HIT_CHANCE
        for t in range(t0, len(hit_log)):
            reward += (hit_log[t] - random_network_hit) * 0.5**(t - t0)

        return reward


#    def __gt__(self, other):
#        if self.fails != other.fails:
#            return self.fails < other.fails
#        elif self.tries != other.tries:
#            return self.tries < other.tries
#        elif self.repeats != other.repeats:
#            return self.repeats < other.repeats
#        else:
#            return self.misses < other.misses

    def __gt__(self, other):
        if isinstance(other, Fitness):
            return self.score > other.score
        else:
            return self.score > other

    def __lt__(self, other):
        if isinstance(other, Fitness):
            return self.score < other.score
        else:
            return self.score < other

    def __add__(self, other):
        if isinstance(other, Fitness):
            return Fitness(
                self.fails + other.fails,
                self.repeats + other.repeats,
                self.tries + other.tries,
                self.misses + other.misses,
                self.hits + other.hits,
                score=self.score + other.score)
        else:
            return self.score + other

    def __radd__(self, other):
        return self.__add__(other)

    def __pow__(self, other):
        return self.score**other

    def __truediv__(self, other):
        return self.score / other

    def __floordiv__(self, other):
        if isinstance(other, Fitness):
            return Fitness(
                self.fails / other.fails,
                self.repeats / other.repeats,
                self.tries / other.tries,
                self.misses / other.misses,
                self.hits / other.hits,
                score=self.score / other.score)
        else:
            return Fitness(
                self.fails / other,
                self.repeats / other,
                self.tries / other,
                self.misses / other,
                self.hits / other,
                score=self.score / other)

    def __str__(self):
        return "{:.3f} fails, {:.3f} repeats, ".format(
            self.fails,
            self.repeats) + "{:.3f} tries, {:.3f} misses == {:.3f}".format(
                self.tries, self.misses, self.score)


class BattleshipTests(unittest.TestCase):
    """TODO: document this."""

    NETWORK_SHAPE = NEURAL_NET_SHAPE
    SHIP_SIZES = SHIP_SIZE_LIST
    # weights go from -10 to 10 (inclusive)
    WEIGHT_REACH = 10
    # each possible weight value differs by 0.001
    # WEIGHT_DIFF = 100

    NUM_FITNESS_TESTS = 4

    GENE_LENGTH = 1
    for s in NETWORK_SHAPE:
        GENE_LENGTH *= s

    @unittest.skip("skipping random_network test")
    def test_random_network(self):

        nn = neuralnet.NeuralNetwork(self.NETWORK_SHAPE)
        genes = neuralnet.flatten(nn.weights)
        fit = self.get_fitness(genes)

        print("fitness: " + str(fit))

    def test(self):
        """TODO: document this."""

        startTime = datetime.datetime.now()

        vis = Vis()

        def fnDisplay(population, gen, info):

            avg = Fitness(0, 0, 0, 0, 0, score=0)
            for ind in population:
                avg = avg + ind.fitness
            avg = avg // len(population)

            max_fit = avg
            for ind in population:
                if max_fit < ind.fitness:
                    max_fit = ind.fitness

            fitness_diversity = np.std(
                [ind.fitness.score for ind in population])

            gene_diversity = np.mean(np.std([[allele for allele in ind.genes]
                             for ind in population], axis=0))

            vis.add_generation(max_fit.score, avg.score,
                               (fitness_diversity, gene_diversity),
                               max_fit.hits, max_fit.misses, info)
            print("Gen " + str(gen) + ":\nAvg Fitness: " + str(avg) +
                  "\nMax Fitness: " + str(max_fit) + "\nElapsed Time: " +
                  str(datetime.datetime.now() - startTime) + "\n -- ")

        def fnGetFitness(genes):
            return self.get_fitness(genes)

        def fnCreate():
            return self.create_gene()

        muts = generate_mutations()
        genetic.evolve(20, [0.29, 0.65, 0.06], 0.11, fnGetFitness, fnDisplay,
                       muts, fnCreate)

        input("END?")

    def create_gene(self):
        """TODO: document this."""
        nn = neuralnet.NeuralNetwork(self.NETWORK_SHAPE)
        genes = neuralnet.flatten(nn.weights)
        return genes

    def get_fitness(self, genes):
        """TODO: document this."""
        weights = neuralnet.unflatten(self.NETWORK_SHAPE, genes)
        network = neuralnet.NeuralNetwork(self.NETWORK_SHAPE, weights=weights)

        result = Result()
        # multithreading inefficient for this
        for i in range(self.NUM_FITNESS_TESTS):
            # running tests
            result = run_game(network, result)

        for i in range(len(result.hit_log)):
            result.hit_log[i] /= self.NUM_FITNESS_TESTS

        return Fitness(
            result.fails / self.NUM_FITNESS_TESTS,
            result.repeats / self.NUM_FITNESS_TESTS,
            result.tries / self.NUM_FITNESS_TESTS / self.NETWORK_SHAPE[0],
            result.misses / self.NUM_FITNESS_TESTS,
            result.hits / self.NUM_FITNESS_TESTS, result.hit_log)


def roulette_select_index(probabilities, invalid):
    sum_total = 0
    for i, prob in enumerate(probabilities):
        if not invalid(i):
            sum_total += prob
        else:
            probabilities[i] = 0
    r = random.uniform(0, sum_total)
    total = 0
    for i, prob in enumerate(probabilities):
        if prob == 0:
            continue
        total += prob
        if r < total:
            return i

    raise Exception(
        "Fell through roulette_select_index -- could be selecting on empty list!"
    )


class Result:
    def __init__(self):
        self.tries = 0
        self.hits = 0
        self.misses = 0
        self.repeats = 0
        self.fails = 0
        self.hit_log = []


def run_game(network, result):
    size = GAME_SIZE
    game = Game(size, ship_sizes=BattleshipTests.SHIP_SIZES)
    # print("board:\n" + str(game.board))
    startTries = result.tries
    time = 0
    while (not game.board.won()
           ) and result.tries < startTries + game.board.squares() * 2.5:
        # input("continue?")
        inputVals = []
        for x in range(size[0]):
            for y in range(size[1]):
                inputVals.append(game.board.get_shot_at((x, y)))
        # print("evaluating on " + str(inputVals))
        selection = list(network.evaluate(inputVals))

        # print("outputs were " + str(selection))
        # select the actual shot by using the probabilities predicted

        def invalid(index):
            if USE_INVALID:
                return game.board.get_shot_at(
                    (index // size[0],
                     index % size[1])) != game.board.UNKNOWN_INT_MARKER
            else:
                return False

        arg = roulette_select_index(selection, invalid)
        shot = (arg // size[0], arg % size[1])
        # print("shooting at " + str(shot))
        if len(result.hit_log) == time:
            result.hit_log.append(0)
        res = game.board.shoot(shot)
        if res is False:
            result.misses += 1
            result.hit_log[time] += 0
        elif res is True:
            result.hits += 1
            result.hit_log[time] += 1
        elif res.startswith("already"):
            result.repeats += 1
        elif res.startswith("invalid"):
            result.fails += 1
        else:
            raise Exception("Invalid return from board.shoot()")
        # print("board: " + str(game.board))
        result.tries += 1
        time += 1
    return result


def generate_mutations():
    def replace(genes):
        # print("replace")
        index = random.randrange(0, len(genes))
        genes[index] = random.uniform(-BattleshipTests.WEIGHT_REACH,
                                      BattleshipTests.WEIGHT_REACH)
        return genes

    def scale(genes):
        # print("scale")
        index = random.randrange(0, len(genes))
        genes[index] *= random.uniform(0.5, 1.5)
        return genes

    def delta_change(genes):
        # print("delta")
        index = random.randrange(0, len(genes))
        change = random.uniform(-1, 1)
        genes[index] += change
        return genes

    def sign_change(genes):
        # print("sign")
        index = random.randrange(0, len(genes))
        genes[index] *= -1
        return genes

    def swap(genes):
        # print("swap")
        first = second = 0
        while (first == second):
            first = random.randrange(0, len(genes))
            second = random.randrange(0, len(genes))
        genes[first], genes[second] = genes[second], genes[first]
        return genes

    # return [replace, scale, delta_change, sign_change, swap]
    return [scale, delta_change, swap]


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

    def hit(self, pos, doDamage=True):
        """Return if the ship is hit by a shot at pos."""

        if pos in self.sectionsAlive:
            if (doDamage):
                self.sectionsAlive[pos] = False
            # print("sectionsAlive: " + str(self.sectionsAlive))
            return True
        else:
            return False


class Board:
    """
    TODO: fully document this.

    shots is a dict with a tuple representing the position pointing to a value,
    1 for hit, 0 for miss. The key doesn't exist if it has not been shot at,
    but get_shot_at returns -1 for no-shot positions.
    """

    HIT_INT_MARKER = 1
    HIT_STR_MARKER = "X"
    MISS_INT_MARKER = 0
    MISS_STR_MARKER = "O"
    UNKNOWN_INT_MARKER = -1
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
            return "invalid position " + str(pos)
        elif pos in self.shots:
            # print("already shot at " + str(pos))
            return "already shot at" + str(pos)
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
        self._ships = list()

        for length in ship_sizes:
            self._ships.append(self.genShip(length))

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

        if xr < 0 or yr < 0:
            raise Exception(
                "Ship size too large, x or y range is negative to fit ship!")

        def overlaps(ship):
            for other_ship in self._ships:
                for this_pos in ship.sectionsAlive.keys():
                    if other_ship.hit(this_pos, False):
                        return True

        while (True):
            pos = (rand(0, xr), rand(0, yr))
            ship = Ship(pos, d * length)
            if not overlaps(ship):
                break

        return ship


def human_play_test():
    """Playtest battleship with a human player (terminal input)."""

    size = tuple(map(int, input("Enter board size (w,h): ").split(",")))
    ships = list(
        map(int,
            input("Enter ship sizes (size1,size2,...)").split(",")))
    game = Game(size, ships)

    print("0, 0 is at the top left")
    while not game.board.won():
        print("board:\n" + str(game.board))

        done = False
        while not done:
            shot = input("Where do you want to shoot (x, y)? ").strip()
            if (shot == "end" or shot == "exit" or shot == "quit"):
                break
            if re.fullmatch(r"\d+[ ]*,[ ]*\d+", shot) is None:
                print("could not parse input, use form 'number, number'!")
            else:
                done = True
        shot = tuple(map(int, shot.split(",")))
        result = game.board.shoot(shot)
        print("Your shot was a " + ("hit!" if result is True else "miss..."
                                    if result is False else "error"))

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
        BattleshipTests().test()
