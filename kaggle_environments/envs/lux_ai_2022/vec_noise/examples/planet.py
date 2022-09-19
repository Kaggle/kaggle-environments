#!/usr/bin/env python
"""Offline procedural planet texture generator using Perlin noise 

A 2D atmospheric texture is generated at startup and stored.
This is then used to shade the planet sphere with simple lighting.

The mouse can be used to rotate the planet axis. Notice the 
pinching distortion near the poles due to the texture wrapping.

requires pyglet 1.1+ and ctypes
"""

import os
import math
import pyglet
from pyglet.gl import *
import ctypes
from vec_noise import pnoise2

TEXTURE_SIZE = 512

def blend(a, c1, c2, multiplier=1.0):
	a = 3*a**2 - 2*a**3
	b = 1.0 - a
	return ((c1[0]*a + c2[0]*b) * multiplier, 
		(c1[1]*a + c2[1]*b) * multiplier, 
		(c1[2]*a + c2[2]*b) * multiplier)

def create_bands_texture(bands=14.0, stretch=2.0, turbulence=8.0, 
	color1=(1.0, 0.8, 0.6), color2=(0.1, -0.3, -0.4)):
	coords = range(TEXTURE_SIZE)
	texel = (ctypes.c_ubyte * (3 * TEXTURE_SIZE**2))()
	for y in coords:
		for x in coords:
			p = pnoise2(x * 15.0 / TEXTURE_SIZE, y * 15.0 / TEXTURE_SIZE, octaves=5, 
				repeatx=15.0, repeaty=15.0) * math.pi * 2.0
			px = (x + math.sin(p) * turbulence)
			py = (y + math.cos(p) * turbulence)
			v = pnoise2(
				px / stretch / TEXTURE_SIZE, py * bands / TEXTURE_SIZE, octaves=4,
				repeaty=bands, repeatx=1.0/stretch)
			r, g, b = blend((v + 1.0) / 2.0, color1, color2, 1.75 + (p**5)*0.003)
			texel[(x + (y * TEXTURE_SIZE))*3] = max(min(int(r * 255.0), 255), 0)
			texel[(x + (y * TEXTURE_SIZE))*3 + 1] = max(min(int(g * 255.0), 255), 0)
			texel[(x + (y * TEXTURE_SIZE))*3 + 2] = max(min(int(b * 255.0), 255), 0)
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, TEXTURE_SIZE, TEXTURE_SIZE, 0, 
		GL_RGB, GL_UNSIGNED_BYTE, ctypes.byref(texel))
	return texel


if __name__ == '__main__':
	import sys
	global xrot, yrot, d
	win = pyglet.window.Window(width=640, height=640, resizable=True, visible=False,
		config=pyglet.gl.Config(sample_buffers=1, samples=4, double_buffer=True, depth_size=24))

	glEnable(GL_LIGHTING)
	glEnable(GL_LIGHT0)
	fourfv = ctypes.c_float * 4
	glLightfv(GL_LIGHT0, GL_POSITION, fourfv(1, 0, 0.5, 0))
	glLightfv(GL_LIGHT0, GL_AMBIENT, fourfv(0.01, 0.01, 0.01, 1.0))
	glLightfv(GL_LIGHT0, GL_DIFFUSE, fourfv(1.5, 1.5, 1.5, 1.5))
	glEnable(GL_COLOR_MATERIAL)

	create_bands_texture()
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
	glEnable(GL_TEXTURE_2D)


	quadratic = gluNewQuadric()
	gluQuadricNormals(quadratic, GLU_SMOOTH)
	gluQuadricTexture(quadratic, GL_TRUE)
	glEnable(GL_CULL_FACE)
	yrot = spin = 0.0
	xrot = 90.0

	def on_resize(width, height):
		glViewport(0, 0, width, height)
		glMatrixMode(GL_PROJECTION)
		glLoadIdentity()
		gluPerspective(70, 1.0*width/height, 0.1, 1000.0)
		glMatrixMode(GL_MODELVIEW)
		glLoadIdentity()
	win.on_resize = on_resize

	@win.event
	def on_mouse_motion(x, y, dx, dy):
		global xrot, yrot
		yrot += dx * 0.3
		xrot += dy * 0.3

	@win.event
	def on_draw():
		global xrot, yrot
		win.clear()
		glLoadIdentity()
		glTranslatef(0, 0, -1.5)
		glRotatef(xrot, 1.0, 0.0, 0.0)
		glRotatef(yrot, 0.0, 1.0, 0.0)
		glRotatef(spin, 0.0, 0.0, 1.0)
		gluSphere(quadratic, 0.65, 60, 60)
		return

		glBegin(GL_QUADS)
		glTexCoord2f(0.0, 0.0)
		glVertex3f(1, -1, 0)
		glTexCoord2f(0.0, 1.0)
		glVertex3f(1, 1, 0)
		glTexCoord2f(1.0, 1.0)
		glVertex3f(-1, 1, 0)
		glTexCoord2f(1.0, 0.0)
		glVertex3f(-1, -1, 0)
		glEnd()

	def update(dt):
		global spin
		spin += dt * 10.0

	pyglet.clock.schedule_interval(update, 1.0/30.0)

	
	win.set_visible()
	win.set_exclusive_mouse()
	pyglet.app.run()
