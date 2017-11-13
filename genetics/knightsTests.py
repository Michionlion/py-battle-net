"""TODO: document this."""


class Position:
    """TODO: document this."""

    def __init__(self, x, y):
        """TODO: document this."""
        self.x = x
        self.y = y


def get_attacks(location, boardWidth, boardHeight):
    """TODO: document this."""
    return [i for i in set(
        Position(x+location.x, y+location.y)
        for x in [-2, -1, 1, 2] if 0 <= x + location.x < boardWidth
        for y in [-2, -1, 1, 2] if 0 <= y + location.y < boardHeight
        and abs(y) != abs(x)
    )]


def create(fnGetRandomPosition, expectedKnights):
    """TODO: document this."""
    genes = [fnGetRandomPosition() for _ in range(expectedKnights)]
    return genes
