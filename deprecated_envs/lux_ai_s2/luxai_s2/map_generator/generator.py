from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.fft import dctn, idctn
from scipy.ndimage import convolve, maximum_filter

from luxai_s2.map_generator.symnoise import SymmetricNoise, symmetrize
from luxai_s2.map_generator.visualize import viz


class GameMap(object):
    def __init__(self, rubble, ice, ore, symmetry):
        self.rubble = rubble
        self.ice = ice
        self.ore = ore
        self.symmetry = symmetry
        self.width, self.height = len(rubble[0]), len(rubble)

    @staticmethod
    def noise(seed=None, noise=None, noise_shift=None, symmetry=None):
        if seed is not None and noise is not None:
            raise ValueError("At most one of seed, noise can be specified.")

        if noise is None:
            noise = SymmetricNoise(seed=seed, symmetry=symmetry, octaves=3)
        else:
            seed = noise.seed
        if noise_shift is not None:
            noise.noise_shift += noise_shift
        if seed is not None and noise_shift is not None:
            noise.random.seed(seed + noise_shift)
        return noise

    @staticmethod
    def random_map(
        seed=None,
        map_type=None,
        map_distribution_type=None,
        symmetry=None,
        width=None,
        height=None,
    ):
        """random_map
        Returns a random map. Can specify seed.
        """
        noise = GameMap.noise(seed)
        if width is None:
            if height is None:
                width = noise.random.randint(32, 64)
            else:
                width = height
        if height is None:
            height = width
        if not map_type:
            map_type = noise.random.choice(["Cave", "Mountain"])
        if map_type == "Cave":
            config_registry = CAVE_CONFIGS
        elif map_type == "Mountain":
            config_registry = MOUNTAIN_CONFIGS
        if not map_distribution_type:
            map_distribution_type = noise.random.choice(
                [
                    "high_ice_high_ore",
                    "high_ice_low_ore",
                    "low_ice_high_ore",
                    "low_ice_low_ore",
                ]
            )
        if not symmetry:
            symmetry = noise.random.choice(
                ["horizontal", "vertical", "rotational", "/", "\\"]
            )
        noise.update_symmetry(symmetry)
        return eval(map_type)(
            width,
            height,
            symmetry,
            noise=noise,
            config=config_registry[map_distribution_type],
        )


@dataclass
class CaveConfig:
    # default parameters -> {'rubble_count': 48.165737847222225, 'ice': 37.93, 'ore': 38.36}
    ice_high_range: Tuple[float, float] = (99, 100)
    ice_mid_range: Tuple[float, float] = (91, 91.7)

    ore_high_range: Tuple[float, float] = (98.7, 100)

    # deeper in walls
    ore_mid_range: Tuple[float, float] = (81, 81.5)


CAVE_CONFIGS = dict(
    # {'rubble_count': 48.165737847222225, 'ice': 37.93, 'ore': 38.36}
    high_ice_high_ore=CaveConfig(),
    low_ice_high_ore=CaveConfig(ice_high_range=(99.7, 100), ice_mid_range=(91, 91.5)),
    high_ice_low_ore=CaveConfig(
        ore_high_range=(99.6, 100),
        ore_mid_range=(81.1, 81.4),
    ),
    # {'rubble_count': 48.165737847222225, 'ice': 16.96, 'ore': 14.57}
    low_ice_low_ore=CaveConfig(
        ore_high_range=(99.6, 100),
        ore_mid_range=(81.1, 81.4),
        ice_high_range=(99.7, 100),
        ice_mid_range=(91, 91.5),
    ),
)


class Cave(GameMap):
    def __init__(
        self,
        width=64,
        height=64,
        symmetry="vertical",
        seed=None,
        noise=None,
        noise_shift=None,
        config: CaveConfig = CAVE_CONFIGS["high_ice_high_ore"],
    ):
        """Cave
        Builds a cave system of size width x height and symmetrical across the `symmetry` axis.

        seed - If noise is not provided, use this seed to generate it.
        noise - A symmetric noise generator.
        noise_shift - Allows use of the same noise function to get different maps.

        Note: The last two are provided as it takes 3x longer to initialize SymmetricNoise than to run.
        """

        noise = GameMap.noise(seed, noise, noise_shift, symmetry)

        """
        Mask will end up with
           0 = interior
           1 = cave wall
           >1 = outside cave
        """
        # Start with mostly zeros
        mask = noise.random.randint(3, size=(height, width))
        mask[mask >= 1] = 1
        # symmetrize(mask, symmetry)

        # Build clumps of ones (will be interior of caves)
        for i in range(3):
            mask = convolve(mask, [[1] * 3] * 3, mode="constant", cval=0) // 6
        mask = 1 - mask

        # Create cave wall
        mask += maximum_filter(mask, size=4)
        # fix bug where filter will cause it to be not symmetric...
        # symmetrize(mask, symmetry)

        # Make some noisy rubble
        x = np.linspace(0, 1, width)
        y = np.linspace(0, 1, height)
        rubble = noise(x, y) * 50 + 50
        rubble = rubble.round()
        rubble[mask == 1] //= 5  # Cave walls
        rubble[mask == 0] = 0  # Interior of cave

        # Make some noisy ice, most ice is on cave edges
        ice = noise(x, y + 100)
        ice[mask > 1] = 0
        ice[mask == 0] = 0
        mid_mask = (ice > np.percentile(ice, config.ice_mid_range[0])) & (
            ice < np.percentile(ice, config.ice_mid_range[1])
        )
        high_mask = (ice > np.percentile(ice, config.ice_high_range[0])) & (
            ice < np.percentile(ice, config.ice_high_range[1])
        )
        ice = mid_mask | high_mask

        # Make some noisy ore, most ore is outside caves
        ore = noise(x, y - 100)
        ore[mask == 1] = 0
        ore[mask == 0] = 0
        mid_mask = (ore > np.percentile(ore, config.ore_mid_range[0])) & (
            ore < np.percentile(ore, config.ore_mid_range[1])
        )
        high_mask = (ore > np.percentile(ore, config.ore_high_range[0])) & (
            ore < np.percentile(ore, config.ore_high_range[1])
        )
        ore = mid_mask | high_mask
        super().__init__(rubble, ice, ore, symmetry)


class Craters(GameMap):
    def __init__(
        self,
        width=64,
        height=64,
        symmetry="vertical",
        seed=None,
        noise=None,
        noise_shift=None,
    ):
        """Craters
        Builds a craters system of size `width` x `height` and symmetrical across the `symmetry` axis.

        seed - If noise is not provided, use this seed to generate it.
        noise - A symmetric noise generator.
        noise_shift - Allows use of the same noise function to get different maps.

        Note: The last two are provided as it takes 3x longer to initialize SymmetricNoise than to run.
        """

        noise = GameMap.noise(seed, noise, noise_shift, symmetry)

        min_craters = max(2, width * height // 1000)
        max_craters = max(4, width * height // 500)
        num_craters = noise.random.randint(min_craters, max_craters + 1)

        # Mask = how many craters have hit the spot. When it symmetrizes, it will divide by 2.
        mask = np.zeros((height, width))
        x = np.linspace(0, 1, width)
        y = np.linspace(0, 1, height)
        crater_noise = (
            noise(x, y, frequency=10) * 0.5 + 0.75
        )  # Don't want perfectly circular craters.
        xx, yy = np.mgrid[:width, :height]

        # ice should be around edges of crater
        ice_mask = np.zeros((height, width))
        for i in range(num_craters):
            cx, cy = noise.random.randint(width), np.random.randint(height)
            cr = noise.random.randint(3, min(width, height) // 4)
            c = (xx - cx) ** 2 + (yy - cy) ** 2
            c = c.T * crater_noise
            mask[c < cr**2] += 1
            edge = np.logical_and(c >= cr**2, c < 2 * cr**2)
            ice_mask[edge] = 2

        mask.astype(float)

        # symmetrize(mask, symmetry)
        # symmetrize(ice_mask, symmetry)

        rubble = (np.minimum(mask, 1) * 90 + 10) * noise(x, y + 100, frequency=3)
        ice = noise(x, y - 100)
        ice[ice_mask == 0] = 0
        ice[ice < np.percentile(ice, 95)] = 0
        ice = np.round(50 * ice + 50)

        ore = np.minimum(mask, 1) * noise(x, y + 100, frequency=3)
        ore[ore < np.percentile(ore, 95)] = 0
        ore = np.round(50 * ore + 50)

        super().__init__(rubble, ice, ore, symmetry)


def solve_poisson(f):
    """
    Solves the Poisson equation
        ∇²p = f
    using the finite difference method with Neumann boundary conditions.
    References: https://elonen.iki.fi/code/misc-notes/neumann-cosine/
                https://scicomp.stackexchange.com/questions/12913/poisson-equation-with-neumann-boundary-conditions
    """
    nx, ny = f.shape

    # Transform to DCT space
    dct = dctn(f, type=1)

    # Divide by magical factors
    cx = np.cos(np.pi * np.arange(nx) / (nx - 1))
    cy = np.cos(np.pi * np.arange(ny) / (ny - 1))
    f = np.add.outer(cx, cy) - 2

    np.divide(dct, f, out=dct, where=f != 0)
    dct[f == 0] = 0

    # Return to normal space
    potential = idctn(dct, type=1)
    return potential / 2


def nabla(x):
    return convolve(x, ((0, 1, 0), (1, -5, 1), (0, 1, 0)))


def dxx(x):
    return convolve(x, ((1,), (-2,), (1,)))


def dyy(x):
    return convolve(x, ((1, -2, 1),))


def dxy(x):
    return convolve(x, ((1, 0, -1), (0, 0, 0), (-1, 0, 1))) / 4


def laplacian(x):
    return convolve(x, ((1, 1, 1), (1, -8, 1), (1, 1, 1))) / 3


@dataclass
class MountainConfig:
    # default parameters -> {'rubble_count': 21.35736545138889, 'ice': 36.19, 'ore': 36.0}

    # controls amount of ice spread along mountain peaks
    ice_high_range: Tuple[float, float] = (98.9, 100)
    # around middle level
    ice_mid_range: Tuple[float, float] = (52.5, 53)
    # around lower level
    ice_low_range: Tuple[float, float] = (0, 21)
    # controls amount of ore spread along middle of the way to the mountain peaks
    ore_mid_range: Tuple[float, float] = (84, 85)
    # controls amount of ore spread along lower part of the mountain peaks, should be smaller than ore_mid_range
    ore_low_range: Tuple[float, float] = (61.4, 62)


MOUNTAIN_CONFIGS = dict(
    # {'rubble_count': 21.35736545138889, 'ice': 36.19, 'ore': 36.0}
    high_ice_high_ore=MountainConfig(),
    # {'rubble_count': 21.35736545138889, 'ice': 16.02, 'ore': 36.0}
    low_ice_high_ore=MountainConfig(
        ice_high_range=(99.5, 100),
        ice_mid_range=(52.8, 53),
        ice_low_range=(0, 20),
    ),
    # {'rubble_count': 21.35736545138889, 'ice': 36.19, 'ore': 16.0}
    high_ice_low_ore=MountainConfig(ore_low_range=(61.7, 62), ore_mid_range=(84.6, 85)),
    # # {'rubble_count': 21.35736545138889, 'ice': 16.02, 'ore': 16.0}
    low_ice_low_ore=MountainConfig(
        ice_high_range=(99.5, 100),
        ice_mid_range=(52.8, 53),
        ice_low_range=(0, 20),
        ore_low_range=(61.7, 62),
        ore_mid_range=(84.6, 85),
    ),
)


class Mountain(GameMap):
    def __init__(
        self,
        width=64,
        height=64,
        symmetry="vertical",
        seed=None,
        noise=None,
        noise_shift=None,
        config: MountainConfig = MOUNTAIN_CONFIGS["high_ice_high_ore"],
    ):
        """Mountain
        Builds a mountain system of size `width` x `height` and symmetrical across the `symmetry` axis.

        seed - If noise is not provided, use this seed to generate it.
        noise - A symmetric noise generator.
        noise_shift - Allows use of the same noise function to get different maps.

        Note: The last two are provided as it takes 3x longer to initialize SymmetricNoise than to run.
        """

        noise = GameMap.noise(seed, noise, noise_shift, symmetry)

        f = np.zeros((height, width))

        # Sprinkle a few mountains on the map.
        min_mountains = max(2, width * height // 750)
        max_mountains = max(4, width * height // 375)
        num_mountains = noise.random.randint(min_mountains, max_mountains + 1)
        for i in range(num_mountains):
            x, y = noise.random.randint(width), noise.random.randint(height)
            f[y][x] -= 1

        # symmetrize(f, symmetry)

        # mask will be floats in [0, 1], where 0 = no mountain, 1 = tallest peak
        mask = solve_poisson(f)
        # symmetrize(mask, symmetry) # in case of floating point errors
        x = np.linspace(0, 1, width)
        y = np.linspace(0, 1, height)
        mask *= 5 + noise(x, y, frequency=3)
        mask -= np.amin(mask)

        # Find the valleys
        Lap = convolve(mask, ((1, 1, 1), (1, -8, 1), (1, 1, 1))) / 3  # Laplacian
        Dxx = convolve(mask, ((1,), (-2,), (1,)))
        Dyy = convolve(mask, ((1, -2, 1),))
        Dxy = convolve(mask, ((1, 0, -1), (0, 0, 0), (-1, 0, 1)))
        det = 16 * Dxx * Dyy - Dxy**2  # Hessian determinant
        det = det.astype(np.csingle)  # complex

        cond = (
            Lap * (2 * Lap - np.sqrt(4 * Lap**2 - det)) / det - 0.25
        )  # ratio of eigenvalues
        cond = abs(cond)  # should already be real except for floating point errors
        cond = np.maximum(cond, 1 / cond)
        # symmetrize(cond, symmetry) # for floating point errors

        def bdry(x, y):
            # Maybe instead of > 20, use some mean and std stuff?
            return abs(cond[y][x]) > 20 and f[y][x] == 0

        closed_set = set()

        def flood_fill(x, y):
            nonlocal closed_set
            region = []
            bounds = []
            open_set = {(x, y)}
            while len(open_set) > 0:
                new_set = set()
                closed_set |= open_set
                for x, y in open_set:
                    region.append((x, y))
                    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        new_x, new_y = x + dx, y + dy
                        if (new_x, new_y) in closed_set:
                            continue
                        if 0 <= new_x < width and 0 <= new_y < height:
                            if bdry(new_x, new_y):
                                bounds.append((new_x, new_y))
                            else:
                                new_set.add((new_x, new_y))
                open_set = new_set
            return region, bounds

        regions = []
        bdrys = []
        for x in range(width):
            for y in range(height):
                if (x, y) in closed_set:
                    continue
                if bdry(x, y):
                    continue
                region, bounds = flood_fill(x, y)
                if len(region) > 0 and len(bounds) > 0:
                    regions.append(region)
                    bdrys.append(bounds)

        for region, bounds in zip(regions, bdrys):
            if len(region) * 5 > width * height:
                continue
            if np.mean(np.take(mask, region)) < 2 * np.mean(np.take(mask, bounds)):
                for x, y in region:
                    mask[y][x] = 0

        for x in range(width):
            for y in range(height):
                if bdry(x, y):
                    mask[y][x] = 0

        mask[np.where(mask > 0)] -= np.amin(mask[np.where(mask > 0)])
        mask -= np.amin(mask)
        mask /= np.amax(mask)

        rubble = (100 * mask).round()
        ice = 100 * mask
        high_mask = (ice > np.percentile(ice, config.ice_high_range[0])) & (
            ice < np.percentile(ice, config.ice_high_range[1])
        )
        mid_mask = (ice > np.percentile(ice, config.ice_mid_range[0])) & (
            ice < np.percentile(ice, config.ice_mid_range[1])
        )
        low_mask = (ice > np.percentile(ice, config.ice_low_range[0])) & (
            ice < np.percentile(ice, config.ice_low_range[1])
        )
        ice = low_mask | mid_mask | high_mask

        ore = 100 * mask

        mid_mask = (ore > np.percentile(ore, config.ore_mid_range[0])) & (
            ore < np.percentile(ore, config.ore_mid_range[1])
        )
        low_mask = (ore > np.percentile(ore, config.ore_low_range[0])) & (
            ore < np.percentile(ore, config.ore_low_range[1])
        )

        ore = low_mask | mid_mask

        super().__init__(rubble, ice, ore, symmetry)


class Island(GameMap):
    def __init__(
        self,
        width=64,
        height=64,
        symmetry="vertical",
        seed=None,
        noise=None,
        noise_shift=None,
    ):
        """Island
        Builds an island system of size `width` x `height` and symmetrical across the `symmetry` axis.

        seed - If noise is not provided, use this seed to generate it.
        noise - A symmetric noise generator.
        noise_shift - Allows use of the same noise function to get different maps.

        Note: The last two are provided as it takes 3x longer to initialize SymmetricNoise than to run.
        """

        noise = GameMap.noise(seed, noise, noise_shift, symmetry)

        # at the end, 0 = island, 1 = sea (of rubble)
        mask = noise.random.randint(4, size=(height, width))
        mask[mask >= 1] = 1

        s = -1
        while np.sum(mask == 0) != s:
            s = np.sum(mask == 0)
            mask = convolve(mask, [[1] * 3] * 3, mode="constant", cval=1) // 6

        # Shift every spot on the map a little, making the islands nosier.
        x = np.linspace(0, 1, width)
        y = np.linspace(0, 1, height)
        noise_dx = noise(x, y, frequency=10) * 4 - 2
        noise_dy = noise(x, y + 100, frequency=10) * 4 - 2

        xx, yy = np.mgrid[:width, :height]
        new_xx = xx + np.round(noise_dx).astype(int)
        new_xx = np.maximum(0, np.minimum(width - 1, new_xx))
        new_yy = yy + np.round(noise_dy).astype(int)
        new_yy = np.maximum(0, np.minimum(height - 1, new_yy))

        mask[yy, xx] = mask[new_yy, new_xx]

        # symmetrize(mask, symmetry)

        rubble = noise(x, y - 100, frequency=3) * 50 + 50
        rubble[mask == 0] //= 20

        # Unsure what to do about ice, ore right now. Place in pockets on islands?
        ice = noise(x, y + 200, frequency=10) ** 2 * 100
        ice[ice < 50] = 0
        ice[mask != 0] = 0

        ore = noise(x, y - 200, frequency=10) ** 2 * 100
        ore[ore < 50] = 0
        ore[mask != 0] = 0

        super().__init__(rubble, ice, ore, symmetry)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate maps for Lux AI S2.")
    parser.add_argument(
        "-t",
        "--map_type",
        help="Map type ('Cave', 'Craters', 'Island', 'Mountain')",
        required=False,
    )
    parser.add_argument("-s", "--size", help="Size (32-64)", required=False)
    parser.add_argument("-d", "--seed", help="Seed")
    parser.add_argument(
        "-m",
        "--symmetry",
        help="Symmetry ('horizontal', 'rotational', 'vertical', '/', '\\')",
    )

    args = vars(parser.parse_args())
    map_type = args.get("map_type", None)
    symmetry = args.get("symmetry", None)
    if args.get("size", None):
        width = height = int(args["size"])
    else:
        width = height = None
    if args.get("seed", None):
        seed = int(args["seed"])
    else:
        seed = None
    game_map = GameMap.random_map(
        seed=seed, symmetry=symmetry, map_type=map_type, width=width, height=height
    )
    viz(game_map)
    input()
