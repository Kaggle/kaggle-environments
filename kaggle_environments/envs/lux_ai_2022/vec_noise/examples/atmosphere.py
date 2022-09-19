#!/usr/bin/env python
"""Real-time atomspheric shader

Utilizes the shader noise functions

requires pyglet 1.1+ and ctypes
"""

import os
import math
import pyglet
from pyglet.gl import *
import ctypes
from vec_noise.shader_noise import ShaderNoiseTexture, shader_noise_glsl
from vec_noise import shader

vert_shader = shader.VertexShader('vertex', '''
	/* simple vertex shader that stores the vertex position, normal
	 * lighting vector and color in varyings for easy access by the 
	 * frag shader
	 */

	uniform float scale;
	varying vec3 position;
	varying vec3 normal;
	varying vec3 lightvec;
	varying vec4 color;

	void main(void) {
		position = gl_Vertex.xyz * scale;
		normal = gl_NormalMatrix * gl_Normal;
		gl_Position = ftransform();
		/* Directional light assumed */
		vec4 v = gl_ModelViewMatrix * gl_Vertex;
		lightvec = gl_LightSource[0].position.xyz - v.xyz;
		color = gl_Color;
	}
''')

atmosphere_frag_shader = shader.FragmentShader('atmosphere', shader_noise_glsl + '''
	/* Animated atmospheric shader */

	uniform float time;
	varying vec3 position;
	varying vec3 normal;
	varying vec3 lightvec;
	varying vec4 color;

	void main(void) {
		float t = fbmnoise(position * 2.0 + time * 0.001, 2);
		vec3 turb = vec3(sin(t * 0.5), cos(t), 0) * 0.04;
		float h = fbmturbulence(turb + position * 0.5 + time * 0.002, 6) * 1.35;
		h = h*h;
		/* Calculate the lighting */
		vec3 N = normalize(normal);
		float intensity = max(0.0, dot(N, normalize(lightvec)));
		float glare = max(0.0, dot(N, normalize(gl_LightSource[0].halfVector.xyz)));
		vec4 ambient = gl_LightSource[0].ambient;
		vec4 diffuse = gl_LightSource[0].diffuse * intensity;
		vec4 specular = gl_LightSource[0].specular * pow(glare, 64.0);
		gl_FragColor = (ambient + diffuse + specular) * color * h;
	}
''')
atmosphere_prog = shader.ShaderProgram(vert_shader, atmosphere_frag_shader)
atmosphere_prog.install()
atmosphere_prog.uset1F('scale', 0.3)


if __name__ == '__main__':
	import sys
	global xrot, yrot, d
	win = pyglet.window.Window(width=640, height=640, resizable=True, visible=False,
		config=pyglet.gl.Config(sample_buffers=1, samples=4, double_buffer=True, depth_size=24))

	glEnable(GL_LIGHTING)
	glEnable(GL_LIGHT0)
	fourfv = ctypes.c_float * 4
	glLightfv(GL_LIGHT0, GL_POSITION, fourfv(1, 0, 1.0, 0.5))
	glLightfv(GL_LIGHT0, GL_AMBIENT, fourfv(0.001, 0.001, 0.001, 1.0))
	glLightfv(GL_LIGHT0, GL_DIFFUSE, fourfv(2.0, 2.0, 2.0, 1.0))
	glLightfv(GL_LIGHT0, GL_SPECULAR, fourfv(0.001, 0.001, 0.001, 1.0))
	glEnable(GL_COLOR_MATERIAL)
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

	noisetex = ShaderNoiseTexture()
	noisetex.load()
	

	earth_texture = pyglet.image.load(
		os.path.join(os.path.dirname(__file__), "earth.1024x512.jpg")).get_mipmapped_texture()
	glTexParameteri(earth_texture.target, GL_TEXTURE_WRAP_S, GL_REPEAT)
	glTexParameteri(earth_texture.target, GL_TEXTURE_WRAP_T, GL_REPEAT)
	glTexParameteri(earth_texture.target, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
	glTexParameteri(earth_texture.target, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
	glEnable(earth_texture.target)

	earth = gluNewQuadric()
	gluQuadricNormals(earth, GLU_SMOOTH)
	gluQuadricTexture(earth, GL_TRUE)
	glEnable(GL_CULL_FACE)
	glColor4f(1, 1, 1, 1)

	atmosphere_depth = 0.01
	atmosphere_speed = 0.85

	yrot = spin = 0.0
	xrot = -90.0
	time = 0

	def on_resize(width, height):
		glViewport(0, 0, width, height)
		glMatrixMode(GL_PROJECTION)
		glLoadIdentity()
		gluPerspective(20, 1.0*width/height, 0.1, 1000.0)
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
		glTranslatef(0, 0, -4.5)
		glRotatef(xrot, 1.0, 0.0, 0.0)
		glRotatef(yrot, 0.0, 1.0, 0.0)
		glRotatef(spin, 0.0, 0.0, 1.0)
		glDisable(GL_TEXTURE_3D)
		glEnable(earth_texture.target)
		glDisable(GL_BLEND)
		gluSphere(earth, 0.65, 60, 60)

		glLoadIdentity()
		glTranslatef(0, 0, -4.5)
		glRotatef(xrot, 1.0, 0.0, 0.0)
		glRotatef(yrot, 0.0, 1.0, 0.0)
		glRotatef(spin * atmosphere_speed, 0.0, 0.0, 1.0)
		glDisable(earth_texture.target)
		noisetex.enable()
		glEnable(GL_BLEND)
		atmosphere_prog.install()
		atmosphere_prog.uset1F('time', time)
		gluSphere(earth, 0.65 + atmosphere_depth, 60, 60)
		atmosphere_prog.uninstall()

	def update(dt):
		global spin, time
		spin += dt * 3.0
		time += dt

	pyglet.clock.schedule_interval(update, 1.0/30.0)

	
	win.set_visible()
	win.set_exclusive_mouse()
	pyglet.app.run()
