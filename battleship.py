import numpy as np
from neuralnet import NeuralNetwork as NN


class Ship:
    def __init__(self, x, length):
        self.length = int(length)
        self.x = int(x)
        self.sectionsAlive = [True] * (length)

        print("length: " + str(self.length))
        print("position: " + str(x))
        print("sections: " + str(self.sectionsAlive))

    #
    def alive(self):
        count = self.length
        for alive in self.sectionsAlive:
            if not alive:
                count -= 1
        #
        if count > 0:
            return True
        else:
            return False
    #

    def hit(self, x):
        x = int(x)
        print("got hit at " + str(x))
        print(self.x <= x)
        print(self.x + self.length)
        print(self.x + self.length > x)
        if self.x <= x and self.x + self.length > x:
            self.sectionsAlive[x - self.x] = False
            print("good hit")
            return True
        else:
            print("bad hit")
            return False
    #

# Ship


class Board:
    def __init__(self, width, ships=[]):
        self.ships = ships
        self.shots = []
        self.misses = []
        self.hits = []
        self.width = width
    #
    # def add_ship(self, ship):
    #    ships.append(ship)
    #

    def shoot(self, x):
        print("shooting with x=" + str(x))
        if x < 0 or x >= self.width:
            print("returning None")
            return None
        else:
            for ship in self.ships:
                print("hitting " + str(ship))
                if ship.hit(x):
                    self.hits.append(x)
                    return True

            self.misses.append(x)
            return False
    #

    def won(self):
        if self.numshots() >= BOARD_SIZE:
            return True
        for ship in self.ships:
            if ship.alive():
                return False
        return True
    #

    def numshots(self):
        return len(self.misses) + len(self.hits)
    #
# Board


if __name__ == '__main__':

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
        if(input == "end"):
            break
        print("Your shot was a " + ("hit!" if board.shoot(int(shot)) else "miss..."))
        print("hits: " + str(board.hits))
        print("misses: " + str(board.misses))
        print("sectionsAlive: " + str(board.ships[0].sectionsAlive))
    #

#
