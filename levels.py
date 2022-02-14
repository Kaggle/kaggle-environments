upgrade_times = [pow(i,2) + 1 for i in range(1, 10)]
SPAWN_VALUES = []
current = 0
for t in upgrade_times:
    current += t
    SPAWN_VALUES.append(current)
print(SPAWN_VALUES)
