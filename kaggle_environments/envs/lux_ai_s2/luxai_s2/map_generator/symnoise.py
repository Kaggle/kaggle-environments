from collections.abc import Iterable

import numpy as np
from vec_noise import snoise2


def symmetrize(x, symmetry="vertical"):
    # In place operation to average along the symmetry.
    height, width = x.shape
    if symmetry == "horizontal":
        x[height // 2 :] += x[(height - 1) // 2 :: -1]
        x[(height - 1) // 2 :: -1] = x[height // 2 :]
    elif symmetry == "vertical":
        x[:, width // 2 :] += x[:, (width - 1) // 2 :: -1]
        x[:, (width - 1) // 2 :: -1] = x[:, width // 2 :]
    elif symmetry == "rotational":
        x[height // 2 - 1 :, :] += x[(height + 1) // 2 :: -1, ::-1]
        x[(height + 1) // 2 :: -1, ::-1] = x[height // 2 - 1 :, :]
    elif symmetry == "/":
        for j in range(height):
            x[j, -j - 1 :: -1] += x[j:, -j - 1]
            x[j:, -j - 1] = x[j, -j - 1 :: -1]
    elif symmetry == "\\":
        for j in range(height):
            x[j, j:] += x[j:, j]
            x[j:, j] = x[j, j:]
    else:
        x *= 2

    if x.dtype.kind == "i":  # Integer arrays need integer division.
        x //= 2
    else:
        x /= 2


class SymmetricNoise(object):
    def __init__(
        self,
        seed: int = 0,
        octaves: int = 1,
        symmetry="vertical",
        width=None,
        height=None,
        noise_shift=0,
    ):
        """Symmetrical Simplex noise.

            ex.: noise = SymmetricalNoise(symmetry="rotational", width=50, height=100,
                                          octaves=3, seed=777)

        Parameters:
            symmetry : one of "vertical", "horizontal", "rotational", "/", and "\\"
            width : width of noise map
            height : height of noise map
            seed : rng seed
            octaves : how fine the features are
        """
        if symmetry not in (None, "vertical", "horizontal", "rotational", "/", "\\"):
            raise ValueError(
                "symmetry must be one of None, 'vertical', 'horizontal', 'rotational', '/', and '\\'"
            )
        if symmetry and symmetry in "/\\" and width != height:
            raise ValueError("width and height must be equal if symmetry = / or \\")

        if not seed:
            seed = np.random.randint(1 << 31)
        self.octaves = octaves
        self.random = np.random.RandomState(seed)
        self.noise_shift = noise_shift
        self.seed = seed

        self.width = width
        self.height = height
        self.symmetry = symmetry

    def update_symmetry(self, symmetry):
        self.symmetry = symmetry

    def noise(self, x=None, y=None, frequency: float = 1):
        # x and y can be arrays, but all values should be between 0 and 1.
        if x is None and self.width is None:
            raise ValueError("Must provide x or define width in initialization")
        if y is None and self.height is None:
            raise ValueError("Must provide y or define height in initialization")

        if x is None:
            x = np.linspace(0, 1, self.width)
        if y is None:
            y = np.linspace(0, 1, self.height)

        x = x + self.noise_shift

        x, y = np.meshgrid(x, y)
        total = snoise2(x, y, octaves=self.octaves)
        symmetrize(total, self.symmetry)
        # Normalize between [0, 1]
        total -= np.amin(total)
        total /= np.amax(total)
        return total

    def __call__(self, *args, **kwargs):
        return self.noise(*args, **kwargs)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    from mpl_toolkits.axes_grid1 import ImageGrid

    SEED = 0
    w, h = 100, 100
    x = np.linspace(0, 1, w)
    y = np.linspace(0, 1, h)
    imgs = []
    for s in ("vertical", "horizontal", "rotational", "/", "\\"):
        noise = SymmetricNoise(symmetry=s, octaves=3, seed=SEED)
        imgs.append(noise(x, y))

    fig = plt.figure(figsize=(10.0, 2.0))
    grid = ImageGrid(
        fig,
        111,
        nrows_ncols=(1, 5),
        axes_pad=0.1,
    )

    for ax, im in zip(grid, imgs):
        # Iterating over the grid returns the Axes.
        ax.imshow(im, cmap="gray")

    plt.show()
