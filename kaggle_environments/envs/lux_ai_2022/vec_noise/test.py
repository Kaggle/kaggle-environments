import unittest


class PerlinTestCase(unittest.TestCase):

    def test_perlin_1d_range(self):
        from vec_noise import pnoise1
        for i in range(-10000, 10000):
            x = i * 0.49
            n = pnoise1(x)
            self.assertTrue(-1.0 <= n <= 1.0, (x, n))

    def test_perlin_1d_octaves_range(self):
        from vec_noise import pnoise1
        for i in range(-1000, 1000):
            for o in range(10):
                x = i * 0.49
                n = pnoise1(x, octaves=o + 1)
                self.assertTrue(-1.0 <= n <= 1.0, (x, n))

    def test_perlin_1d_base(self):
        from vec_noise import pnoise1
        self.assertEqual(pnoise1(0.5), pnoise1(0.5, base=0))
        self.assertNotEqual(pnoise1(0.5), pnoise1(0.5, base=5))
        self.assertNotEqual(pnoise1(0.5, base=5), pnoise1(0.5, base=1))

    def test_perlin_2d_range(self):
        from vec_noise import pnoise2
        for i in range(-10000, 10000):
            x = i * 0.49
            y = -i * 0.67
            n = pnoise2(x, y)
            self.assertTrue(-1.0 <= n <= 1.0, (x, y, n))

    def test_perlin_2d_octaves_range(self):
        from vec_noise import pnoise2
        for i in range(-1000, 1000):
            for o in range(10):
                x = -i * 0.49
                y = i * 0.67
                n = pnoise2(x, y, octaves=o + 1)
                self.assertTrue(-1.0 <= n <= 1.0, (x, n))

    def test_perlin_2d_base(self):
        from vec_noise import pnoise2
        x, y = 0.73, 0.27
        self.assertEqual(pnoise2(x, y), pnoise2(x, y, base=0))
        self.assertNotEqual(pnoise2(x, y), pnoise2(x, y, base=5))
        self.assertNotEqual(pnoise2(x, y, base=5), pnoise2(x, y, base=1))

    def test_perlin_3d_range(self):
        from vec_noise import pnoise3
        for i in range(-10000, 10000):
            x = -i * 0.49
            y = i * 0.67
            z = -i * 0.727
            n = pnoise3(x, y, z)
            self.assertTrue(-1.0 <= n <= 1.0, (x, y, z, n))

    def test_perlin_3d_octaves_range(self):
        from vec_noise import pnoise3
        for i in range(-1000, 1000):
            x = i * 0.22
            y = -i * 0.77
            z = -i * 0.17
            for o in range(10):
                n = pnoise3(x, y, z, octaves=o + 1)
                self.assertTrue(-1.0 <= n <= 1.0, (x, y, z, n))

    def test_perlin_3d_base(self):
        from vec_noise import pnoise3
        x, y, z = 0.1, 0.7, 0.33
        self.assertEqual(pnoise3(x, y, z), pnoise3(x, y, z, base=0))
        self.assertNotEqual(pnoise3(x, y, z), pnoise3(x, y, z, base=5))
        self.assertNotEqual(pnoise3(x, y, z, base=5), pnoise3(x, y, z, base=1))


class SimplexTestCase(unittest.TestCase):

    def test_simplex_2d_range(self):
        from vec_noise import snoise2
        for i in range(-10000, 10000):
            x = i * 0.49
            y = -i * 0.67
            n = snoise2(x, y)
            self.assertTrue(-1.0 <= n <= 1.0, (x, y, n))

    def test_simplex_2d_octaves_range(self):
        from vec_noise import snoise2
        for i in range(-1000, 1000):
            for o in range(10):
                x = -i * 0.49
                y = i * 0.67
                n = snoise2(x, y, octaves=o + 1)
                self.assertTrue(-1.0 <= n <= 1.0, (x, n))

    def test_simplex_3d_range(self):
        from vec_noise import snoise3
        for i in range(-10000, 10000):
            x = i * 0.31
            y = -i * 0.7
            z = i * 0.19
            n = snoise3(x, y, z)
            self.assertTrue(-1.0 <= n <= 1.0, (x, y, z, n))

    def test_simplex_3d_octaves_range(self):
        from vec_noise import snoise3
        for i in range(-1000, 1000):
            x = -i * 0.12
            y = i * 0.55
            z = i * 0.34
            for o in range(10):
                n = snoise3(x, y, z, octaves=o + 1)
                self.assertTrue(-1.0 <= n <= 1.0, (x, y, z, o+1, n))

    def test_simplex_4d_range(self):
        from vec_noise import snoise4
        for i in range(-10000, 10000):
            x = i * 0.88
            y = -i * 0.11
            z = -i * 0.57
            w = i * 0.666
            n = snoise4(x, y, z, w)
            self.assertTrue(-1.0 <= n <= 1.0, (x, y, z, w, n))

    def test_simplex_4d_octaves_range(self):
        from vec_noise import snoise4
        for i in range(-1000, 1000):
            x = -i * 0.12
            y = i * 0.55
            z = i * 0.34
            w = i * 0.21
            for o in range(10):
                n = snoise4(x, y, z, w, octaves=o + 1)
                self.assertTrue(-1.0 <= n <= 1.0, (x, y, z, w, o+1, n))


if __name__ == '__main__':
    unittest.main()
