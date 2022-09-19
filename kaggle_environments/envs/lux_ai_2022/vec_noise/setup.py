import sys

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext

class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())

if sys.platform != 'win32':
    compile_args = ['-funroll-loops']
else:
    # XXX insert win32 flag to unroll loops here
    compile_args = []

setup(
    name='vec_noise',
    version='1.1.4',
    description='Vectorized Perlin noise for Python',
    long_description='''\
This is a fork of Casey Duncan's noise library that vectorizes all of the noise
functions using NumPy. It is much faster than the original for computing noise
values at many coordinates.

Perlin noise is ubiquitous in modern CGI. Used for procedural texturing,
animation, and enhancing realism, Perlin noise has been called the "salt" of
procedural content. Perlin noise is a type of gradient noise, smoothly
interpolating across a pseudo-random matrix of values.

The vec_noise library includes native-code implementations of Perlin "improved"
noise and Perlin simplex noise. It also includes a fast implementation of
Perlin noise in GLSL, for use in OpenGL shaders. The shader code and many of
the included examples require Pyglet (http://www.pyglet.org), the native-code
noise functions themselves do not, however.

The Perlin improved noise functions can also generate fBm (fractal Brownian
motion) noise by combining multiple octaves of Perlin noise. Shader functions
for convenient generation of turbulent noise are also included.
''',
    author='Zev Benjamin',
    author_email='zev@strangersgate.com',
    url='https://github.com/zbenjamin/vec_noise',
    classifiers = [
        'Development Status :: 4 - Beta',
        'Topic :: Multimedia :: Graphics',
        'License :: OSI Approved :: MIT License',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Programming Language :: C',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
    ],

    package_dir={'vec_noise': ''},
    packages=['vec_noise'],
    setup_requires=['numpy'],
    install_requires=['numpy'],
    cmdclass={'build_ext': build_ext},
    ext_modules=[
        Extension('vec_noise._simplex', ['_simplex.c'],
            extra_compile_args=compile_args,
        ),
        Extension('vec_noise._perlin', ['_perlin.c'],
            extra_compile_args=compile_args,
        )
    ],
)
