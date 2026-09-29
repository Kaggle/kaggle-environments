import numpy as np

# Inline port of vec_noise.snoise2 (2D simplex noise with fBm), which is unseeded.
# fmt: off
_PERM = np.array([
    151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194, 233, 7, 225, 140, 36, 103, 30, 69, 142,
    8, 99, 37, 240, 21, 10, 23, 190, 6, 148, 247, 120, 234, 75, 0, 26, 197, 62, 94, 252, 219, 203,
    117, 35, 11, 32, 57, 177, 33, 88, 237, 149, 56, 87, 174, 20, 125, 136, 171, 168, 68, 175, 74, 165,
    71, 134, 139, 48, 27, 166, 77, 146, 158, 231, 83, 111, 229, 122, 60, 211, 133, 230, 220, 105, 92, 41,
    55, 46, 245, 40, 244, 102, 143, 54, 65, 25, 63, 161, 1, 216, 80, 73, 209, 76, 132, 187, 208, 89,
    18, 169, 200, 196, 135, 130, 116, 188, 159, 86, 164, 100, 109, 198, 173, 186, 3, 64, 52, 217, 226, 250,
    124, 123, 5, 202, 38, 147, 118, 126, 255, 82, 85, 212, 207, 206, 59, 227, 47, 16, 58, 17, 182, 189,
    28, 42, 223, 183, 170, 213, 119, 248, 152, 2, 44, 154, 163, 70, 221, 153, 101, 155, 167, 43, 172, 9,
    129, 22, 39, 253, 19, 98, 108, 110, 79, 113, 224, 232, 178, 185, 112, 104, 218, 246, 97, 228, 251, 34,
    242, 193, 238, 210, 144, 12, 191, 179, 162, 241, 81, 51, 145, 235, 249, 14, 239, 107, 49, 192, 214, 31,
    181, 199, 106, 157, 184, 84, 204, 176, 115, 121, 50, 45, 127, 4, 150, 254, 138, 236, 205, 93, 222, 114,
    67, 29, 24, 72, 243, 141, 128, 195, 78, 66, 215, 61, 156, 180,
] * 2)
_GRAD = np.array([
    (1, 1), (-1, 1), (1, -1), (-1, -1), (1, 0), (-1, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (0, 1), (0, -1),
], dtype=float)
# fmt: on
_F2 = 0.5 * (np.sqrt(3.0) - 1.0)
_G2 = (3.0 - np.sqrt(3.0)) / 6.0


def _simplex2(x, y):
    s = (x + y) * _F2
    i = np.floor(x + s)
    j = np.floor(y + s)
    t = (i + j) * _G2
    x0 = x - (i - t)
    y0 = y - (j - t)
    i1 = (x0 > y0).astype(int)
    j1 = 1 - i1
    corners = (
        (x0, y0, 0, 0),
        (x0 - i1 + _G2, y0 - j1 + _G2, i1, j1),
        (x0 + 2 * _G2 - 1, y0 + 2 * _G2 - 1, 1, 1),
    )
    ii = i.astype(int) & 255
    jj = j.astype(int) & 255
    total = np.zeros_like(x0)
    for cx, cy, di, dj in corners:
        g = _GRAD[_PERM[ii + di + _PERM[jj + dj]] % 12]
        f = np.maximum(0.5 - cx * cx - cy * cy, 0)
        total += f**4 * (g[..., 0] * cx + g[..., 1] * cy)
    return 70 * total


def snoise2(x, y, octaves=1, persistence=0.5, lacunarity=2.0):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    total = np.zeros(np.broadcast(x, y).shape)
    freq, amp, max_amp = 1.0, 1.0, 0.0
    for _ in range(octaves):
        total += _simplex2(x * freq, y * freq) * amp
        max_amp += amp
        freq *= lacunarity
        amp *= persistence
    return total / max_amp


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
            raise ValueError("symmetry must be one of None, 'vertical', 'horizontal', 'rotational', '/', and '\\'")
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
