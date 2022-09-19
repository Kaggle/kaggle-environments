"""shader_noise shader function and texture generator
as described in "GPU Gems" chapter 5:

http://http.developer.nvidia.com/GPUGems/gpugems_ch05.html
"""

__version__ = "$Id: shader_noise.py 37 2008-06-27 22:25:39Z casey.duncan $"

from vec_noise import pnoise3
import ctypes
from pyglet.gl import *

class ShaderNoiseTexture:
	"""tiling 3D noise texture with two channels for use by the
	shader noise functions.
	"""

	def __init__(self, freq=8, width=32):
		"""Generate the 3D noise texture.

		freq -- frequency of generated noise over the width of the 
		texture.

		width -- Width of the texture in texels. The texture is cubic,
		thus all sides are the same width. Must be a power of two.
		Using a larger width can reduce artifacts caused by linear
		interpolation of the noise texture, at the cost of video
		memory, and possibly slower texture access.
		"""
		self.freq = freq
		self.width = width
		scale = float(freq) / width
		width2 = width**2
		texel = (ctypes.c_ushort * (2 * width**3))()
		for z in range(width):
			for y in range(width):
				for x in range(width):
					texel[(x + (y * width) + (z * width2)) * 2] = int((pnoise3(
						x * scale, y * scale, z * scale, 
						repeatx=freq, repeaty=freq, repeatz=freq) + 1.0) * 32767)
					texel[(x + (y * width) + (z * width2)) * 2 + 1] = int((pnoise3(
						x * scale, y * scale, z * scale, 
						repeatx=freq, repeaty=freq, repeatz=freq, base=freq + 1) + 1.0) * 32767)
		self.data = texel
	
	def load(self):
		"""Load the noise texture data into the current texture unit"""
		glTexImage3D(GL_TEXTURE_3D, 0, GL_LUMINANCE16_ALPHA16, 
			self.width, self.width, self.width, 0, GL_LUMINANCE_ALPHA, 
			GL_UNSIGNED_SHORT, ctypes.byref(self.data))
	
	def enable(self):
		"""Convenience method to enable 3D texturing state so the texture may be used by the 
		ffpnoise shader function
		"""
		glEnable(GL_TEXTURE_3D)
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_REPEAT)
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_REPEAT)
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_REPEAT)
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)


shader_noise_glsl = '''
/*
 * GLSL Shader functions for fast fake Perlin 3D noise
 *
 * The required shader_noise_tex texture can be generated using the
 * ShaderNoiseTexture class.  It is a toroidal tiling 3D texture with each texel
 * containing two 16-bit noise source channels. The shader permutes the source
 * texture values by combining the channels such that the noise repeats at a
 * much larger interval than the input texture.
 */

uniform sampler3D shader_noise_tex;
const float twopi = 3.1415926 * 2.0;

/* Simple perlin noise work-alike */
float
pnoise(vec3 position)
{
	vec4 hi = 2.0 * texture3D(shader_noise_tex, position.xyz) - 1.0;
	vec4 lo = 2.0 * texture3D(shader_noise_tex, position.xyz / 9.0) - 1.0;
	return hi.r * cos(twopi * lo.r) + hi.a * sin(twopi * lo.r);
}

/* Multi-octave fractal brownian motion perlin noise */
float
fbmnoise(vec3 position, int octaves)
{
	float m = 1.0;
	vec3 p = position;
	vec4 hi = vec4(0.0);
	/* XXX Loops may not work correctly on all video cards */
	for (int x = 0; x < octaves; x++) {
		hi += (2.0 * texture3D(shader_noise_tex, p.xyz) - 1.0) * m;
		p *= 2.0;
		m *= 0.5;
	}
	vec4 lo = 2.0 * texture3D(shader_noise_tex, position.xyz / 9.0) - 1.0;
	return hi.r * cos(twopi * lo.r) + hi.a * sin(twopi * lo.r);
}

/* Multi-octave turbulent noise */
float
fbmturbulence(vec3 position, int octaves)
{
	float m = 1.0;
	vec3 p = position;
	vec4 hi = vec4(0.0);
	/* XXX Loops may not work correctly on all video cards */
	for (int x = 0; x < octaves; x++) {
		hi += abs(2.0 * texture3D(shader_noise_tex, p.xyz) - 1.0) * m;
		p *= 2.0;
		m *= 0.5;
	}
	vec4 lo = texture3D(shader_noise_tex, position.xyz / 9.0);
	return 2.0 * mix(hi.r, hi.a, cos(twopi * lo.r) * 0.5 + 0.5) - 1.0;
}

'''

if __name__ == '__main__':
	# Demo using a simple noise-textured rotating sphere
	import shader
	win = pyglet.window.Window(width=640, height=640, resizable=True, visible=False)
	vert_shader = shader.VertexShader('stupid', '''
		/* simple vertex shader that stores the vertex position in a varying 
		 * for easy access by the frag shader
		 */
		varying vec3 position;

		void main(void) {
			position = gl_Vertex.xyz * 5.0;
			gl_Position = ftransform();
		}
	''')
	frag_shader = shader.FragmentShader('noise_test', shader_noise_glsl + '''
		varying vec3 position;

		void main(void) {
			float v;
			float a = atan(position.y, position.x);
			float arc = 3.14159 / 3.0;
			if (a > -arc && a < arc) {
				v = pnoise(position) * 0.5 + 0.5;
			} else if (a > arc && a < arc * 4.0) {
				v = fbmnoise(position, 4) * 0.5 + 0.5;
			} else {
				v = fbmturbulence(position, 4) * 0.5 + 0.5;
			}
			gl_FragColor = vec4(v, v, v, 1.0);
		}
	''')
	shader_prog = shader.ShaderProgram(vert_shader, frag_shader)
	shader_prog.install()
	tex = ShaderNoiseTexture()
	tex.load()
	tex.enable()
	shader_prog.uset1I('shader_noise_tex', 0)

	quadratic = gluNewQuadric()
	gluQuadricNormals(quadratic, GLU_SMOOTH)
	gluQuadricTexture(quadratic, GL_TRUE)
	glEnable(GL_CULL_FACE)
	global spin
	spin = 0

	def on_resize(width, height):
		glViewport(0, 0, width, height)
		glMatrixMode(GL_PROJECTION)
		glLoadIdentity()
		gluPerspective(70, 1.0*width/height, 0.1, 1000.0)
		glMatrixMode(GL_MODELVIEW)
		glLoadIdentity()
	win.on_resize = on_resize

	@win.event
	def on_draw():
		global spin
		win.clear()
		glLoadIdentity()
		glTranslatef(0, 0, -1.5)
		glRotatef(spin, 1.0, 1.0, 1.0)
		gluSphere(quadratic, 0.65, 60, 60)

	def update(dt):
		global spin
		spin += dt * 10.0
	pyglet.clock.schedule_interval(update, 1.0/30.0)

	win.set_visible()
	pyglet.app.run()

