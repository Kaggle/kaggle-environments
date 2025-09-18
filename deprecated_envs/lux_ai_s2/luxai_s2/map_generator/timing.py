import timeit

N = 100
for map_type in ["Cave", "Craters", "Island", "Mountain"]:
    time = timeit.timeit(
        f"{map_type}(64, 64)", setup=f"from generator import {map_type}", number=N
    )
    print(f"Took {time / N * 1000:.2f}ms for {map_type}.")


# If you want to use cProfile, you can look in git history.
