import sys
import pyglet
from pyglet.gl import *
from vec_noise import pnoise1
window = pyglet.window.Window(visible=False, resizable=True)

def on_resize(width, height):
	"""Setup 3D viewport"""
	glViewport(0, 0, width, height)
	glMatrixMode(GL_PROJECTION)
	glLoadIdentity()
	gluPerspective(70, 1.0*width/height, 0.1, 1000.0)
	glMatrixMode(GL_MODELVIEW)
	glLoadIdentity()
window.on_resize = on_resize
window.set_visible()

points = 256
span = 5.0
speed = 1.0

if len(sys.argv) > 1:
	octaves = int(sys.argv[1])
else:
	octaves = 1

base = 0
min = max = 0

@window.event
def on_draw():
	global min,max
	window.clear()
	glLoadIdentity()
	glTranslatef(0, 0, -1)
	r = range(256)
	glBegin(GL_LINE_STRIP)
	for i in r:
		x = float(i) * span / points - 0.5 * span
		y = pnoise1(x + base, octaves)
		glVertex3f(x * 2.0 / span, y, 0)
	glEnd()

def update(dt):
	global base
	base += dt * speed
pyglet.clock.schedule_interval(update, 1.0/30.0)

pyglet.app.run()
