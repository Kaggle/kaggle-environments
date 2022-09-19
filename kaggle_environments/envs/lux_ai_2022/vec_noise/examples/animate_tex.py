#!/usr/bin/env python
"""Famous OpenGL teapot renderer for testing textures

requires pyglet 1.1+ and ctypes
"""

import os
import math
import pyglet
from pyglet.gl import *
import ctypes
import vec_noise
from vec_noise import pnoise3, snoise3

def create_3d_texture(width, scale):
	"""Create a grayscale 3d texture map with the specified 
	pixel width on each side and load it into the current texture
	unit. The luminace of each texel is derived using the input 
	function as:

	v = func(x * scale, y * scale, z * scale)

	where x, y, z = 0 in the center texel of the texture.
	func(x, y, z) is assumed to always return a value in the 
	range [-1, 1].
	"""
	coords = range(width)
	texel = (ctypes.c_byte * width**3)()
	half = 0 #width * scale / 2.0 

	for z in coords:
		for y in coords:
			for x in coords:
				v = snoise3(x * scale - half, y * scale - half, z * scale - half, octaves=4, persistence=0.25)
				texel[x + (y * width) + (z * width**2)] = int(v * 127.0)
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
	glTexImage3D(GL_TEXTURE_3D, 0, GL_LUMINANCE, width, width, width, 0, 
		GL_LUMINANCE, GL_BYTE, ctypes.byref(texel))
	return texel
	

if __name__ == '__main__':
	import sys
	global xrot, yrot, d
	win = pyglet.window.Window(width=320, height=320, resizable=True, visible=False,
		config=pyglet.gl.Config(sample_buffers=1, samples=4, double_buffer=True, depth_size=24))

	#noise.randomize()
	create_3d_texture(128, 1/48.0)
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_REPEAT)
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_REPEAT)
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
	glEnable(GL_TEXTURE_3D)
	xrot = yrot = d = 0

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
		global xrot, yrot, d
		glClear(GL_COLOR_BUFFER_BIT)
		glLoadIdentity()
		glTranslatef(0, 0, -1.5)
		glRotatef(xrot, 1.0, 0.0, 0.0)
		glRotatef(yrot, 0.0, 1.0, 0.0)
		glBegin(GL_QUADS)
		glTexCoord3f(0.0, 0.0, d)
		glVertex3f(1, -1, 0)
		glTexCoord3f(0.0, 1.0, d)
		glVertex3f(1, 1, 0)
		glTexCoord3f(1.0, 1.0, d)
		glVertex3f(-1, 1, 0)
		glTexCoord3f(1.0, 0.0, d)
		glVertex3f(-1, -1, 0)
		glEnd()
	
	def update(dt):
		global d
		d += dt * 0.1
		if d > 1.0:
			d -= 1.0

	pyglet.clock.schedule_interval(update, 1/30.0)
	
	win.set_visible()
	win.set_exclusive_mouse()
	pyglet.app.run()

