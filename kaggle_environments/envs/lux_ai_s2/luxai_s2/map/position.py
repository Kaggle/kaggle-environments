import numpy as np


class Position:
    def __init__(self, pos: np.ndarray) -> None:
        self.pos: np.ndarray = pos

    @property
    def x(self) -> int:
        return self.pos[0]

    @property
    def y(self) -> int:
        return self.pos[1]

    def __add__(self, o):
        if isinstance(o, Position):
            return Position(pos=self.pos + o.pos)
        else:
            return Position(pos=self.pos + o)

    def __sub__(self, o):
        if isinstance(o, Position):
            return Position(pos=self.pos - o.pos)
        else:
            return Position(pos=self.pos - o)

    def __str__(self) -> str:
        return f"({self.x}, {self.y})"
