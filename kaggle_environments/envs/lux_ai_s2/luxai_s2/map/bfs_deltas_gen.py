import numpy as np

move_deltas = np.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]])


def gen_deltas(size=3):
    start = (0, 0)
    f = [start]
    seen = set()
    r = set()
    while len(f) > 0:
        pos = f.pop(0)
        seen.add(pos)
        if abs(pos[0]) + abs(pos[1]) > size:
            break
        r.add(pos)
        for md in move_deltas:
            new_pos = (pos[0] + md[0], pos[1] + md[1])
            if new_pos not in seen:
                f.append(new_pos)
    print(r)


gen_deltas(6)
