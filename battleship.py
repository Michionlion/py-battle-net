
class Ship:
    def __init__(self, x, y, width, height):
        self.width = width
        self.height = height
        self.x = x
        self.y = y
        self.sectionsAlive = True * (width*height)
    #
    
    def alive(self):
        return True
    #
    
    def collides(self, x, y):
        if s.x >= x and s.x <= x + s.width:
            if s.y >= y and s.y <= y + s.height:
                return True
    #
    
# Ship

class Board:

    def __init__(self, ships=[]):
        self.ships = ships
        self.shots = []
        self.misses = []
        self.hits = []
    #
    
    def add_ship(self, ship):
        ships.append(ship)
    #
    
    def shoot(self, x, y):
        for s in ships:
            if s.hit(x, y):
                hits.append((x,y))
            #
        #
    #
    
# Board











if __name__ == '__main__':
    
    board = Board()
    
    
    
    
#