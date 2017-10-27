"""TODO: document this."""
import numpy as np
import neuralnet
import unittest
# from genetics import genetic


class BattleshipTests(unittest.TestCase):
    """TODO: document this."""

    NETWORK_SHAPE = (8, 12, 8)
    # weights go from -10 to 10 (inclusive)
    WEIGHT_REACH = 10
    # each possible weight value differs by 0.001
    WEIGHT_DIFF = 100

    NUM_FITNESS_TESTS = 7

    def test(self):
        """TODO: document this."""
        # geneset = [
        #    i / (self.WEIGHT_REACH * self.WEIGHT_DIFF)
        #    for i in range(-self.WEIGHT_DIFF *
        #                   self.WEIGHT_REACH**2,
        #                   self.WEIGHT_DIFF *
        #                   self.WEIGHT_REACH**2 + 1)]
        # print(geneset)
        nn = neuralnet.NeuralNetwork(self.NETWORK_SHAPE)
        genes = neuralnet.flatten(nn.weights)
        self.get_fitness(genes)

    def get_fitness(self, genes):
        """TODO: document this."""
        weights = neuralnet.unflatten(self.NETWORK_SHAPE, genes)
        network = neuralnet.NeuralNetwork(self.NETWORK_SHAPE, weights=weights)
        fitness = 0
        for i in range(self.NUM_FITNESS_TESTS):
            game = Game()
            while not game.board.won():

                inputVals = []
                for x in range(self.NETWORK_SHAPE[0]):
                    if ((x, 1), 1) in game.board.shots:
                        inputVals.append(1)
                    elif ((x, 1), -1) in game.board.shots:
                        inputVals.append(-1)
                    else:
                        inputVals.append(0)
                #
                print("evaluating on " + str(inputVals))
                selection = network.evaluate(inputVals)
                shot = np.argmin(selection)
                print("shooting at " + str((shot, 1)))
                result = game.board.shoot(int(shot))
                if result:
                    print("That was a hit!")
                else:
                    print("That was a miss...")
                    fitness += 1
                print("board: " + str(board))
        #
        return fitness / self.NUM_FITNESS_TESTS
#


class Ship:
    """TODO: document this."""

    def __init__(self, x, length):
        """TODO: document this."""
        self.length = int(length)
        self.x = int(x)
        self.sectionsAlive = [True] * (length)

        print("length: " + str(self.length))
        print("position: " + str(x))
        print("sections: " + str(self.sectionsAlive))

    #
    def alive(self):
        """TODO: document this."""
        count = self.length
        for alive in self.sectionsAlive:
            if not alive:
                count -= 1
        #
                if(input == "end"):
                    break
        if count > 0:
            return True
        else:
            return False
    #

    def hit(self, x):
        """TODO: document this."""
        x = int(x)
        print("got hit at " + str(x))
        print(self.x <= x)
        print(self.x + self.length)
        print(self.x + self.length > x)
        if self.x <= x and self.x + self.length > x:
            self.sectionsAlive[x - self.x] = False
            print("good hit")
            return True
            return True
        else:
            print("bad hit")
            return False
    #

# Shipwidth


class Board:
    """TODO: document this."""

    def __init__(self, size, ships=[]):
        """TODO: document this."""
        self.ships = ships
        self.shots = []
        self.size = size
    #

    def shoot(self, x):
        """TODO: document this."""
        print("shooting with x=" + str(x))
        if x < 0 or x >= self.size[0]:
            print("returning None")
            return None
        else:
            for ship in self.ships:
                print("hitting " + str(ship))
                if ship.hit(x):
                    self.shots.append(((x, 1), 1))
                    return True

            self.shots.append(((x, 1), -1))
            return False
    #

    def won(self):
        """TODO: document this."""
        if self.numshots() >= self.size[0]*self.size[1]:
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
            if not y == 0:
                rep = rep[:-1]  # remove last space
                rep += "\n"
            for x in range(self.size[0]):
                if ((x, y), 1) in self.shots:
                    rep += "X "
                elif ((x, y), -1) in self.shots:
                    rep += "O "
                else:
                    rep += ". "

        rep = rep[:-1]  # remove last space
        return rep
# Board


class Game:
    """TODO: document this."""

    SHIP_SIZES = [3]  # , 2, 1]

    def __init__(self, size=(8, 1)):
        """TODO: document this."""
        self._board_size = size
        self._ships = [Ship(np.random.randint(0, size[0] - length + 1), length)
                       for length in range(len(self.SHIP_SIZES))]
        self.board = Board(size, self._ships)

#


if __name__ == '__main__':
    ans = input("test or play? (t/p) ")
    if ans == "t":
        unittest.main()
    elif ans == "p":
        BOARD_SIZE = 10
        SHIP_SIZES = [3]  # , 2, 1]
        NUM_SHIPS = 1

        ships = []
        for i in SHIP_SIZES:
            x = np.random.randint(0, BOARD_SIZE - i + 1)
            # need to check against other ships for collisions
            ships.append(Ship(x, i))
        #
        board = Board(BOARD_SIZE, ships)

        while not board.won():
            shot = input("Where do you want to shoot? ")
            if(shot == "end"):
                break
            print("Your shot was a " +
                  ("hit!" if board.shoot(int(shot)) else "miss..."))
            print("hits: " + str(board.hits))
            print("misses: " + str(board.misses))
            print("sectionsAlive: " + str(board.ships[0].sectionsAlive))
        #

#
