#!/usr/bin/env python
"""Bump texturing using pixel shaders
using techniques inspired by "GPU Gems" chapter 5:

http://http.developer.nvidia.com/GPUGems/gpugems_ch05.html

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
		position = gl_Vertex.xyz / scale;
		normal = gl_NormalMatrix * gl_Normal;
		gl_Position = ftransform();
		/* Directional light assumed */
		vec4 v = gl_ModelViewMatrix * gl_Vertex;
		lightvec = gl_LightSource[0].position.xyz - v.xyz;
		color = gl_Color;
	}
''')

smooth_frag_shader = shader.FragmentShader('smooth', '''
	/* Simple per-pixel lighting shader */

	uniform float height;
	varying vec3 position;
	varying vec3 normal;
	varying vec3 lightvec;
	varying vec4 color;

	void main(void) {
		vec3 N = normalize(normal);

		/* Calculate the lighting */
		float intensity = max(0.0, dot(N, normalize(lightvec)));
		float glare = max(0.0, dot(N, normalize(gl_LightSource[0].halfVector.xyz)));
		vec4 ambient = gl_LightSource[0].ambient;
		vec4 diffuse = gl_LightSource[0].diffuse * intensity;
		vec4 specular = gl_LightSource[0].specular * pow(glare, 128.0);
		gl_FragColor = (ambient + diffuse + specular) * color;
	}
''')
smooth_prog = shader.ShaderProgram(vert_shader, smooth_frag_shader)

simple_bump_frag_shader = shader.FragmentShader('simple_bump', shader_noise_glsl + '''
	/* Simple normal mapper using perlin noise */

	uniform float height;
	varying vec3 position;
	varying vec3 normal;
	varying vec3 lightvec;
	varying vec4 color;

	void main(void) {
		const float offset = 0.05; // offset for computing surface derivative

		/* Calculate the normal resulting from the bump surface */
		float h = pnoise(position);
		float f0 = h * height;
		float fx = pnoise(position + vec3(offset, 0.0, 0.0)) * height;
		float fy = pnoise(position + vec3(0.0, offset, 0.0)) * height;
		float fz = pnoise(position + vec3(0.0, 0.0, offset)) * height;
		vec3 df = vec3((fx - f0) / offset, (fy - f0) / offset, (fz - f0) / offset);
		vec3 N = normalize(normal - df);

		/* Calculate the lighting */
		float intensity = max(0.0, dot(N, normalize(lightvec)));
		float glare = max(0.0, dot(N, normalize(gl_LightSource[0].halfVector.xyz)));
		h = (h + 1.0) * 0.5;
		vec4 ambient = gl_LightSource[0].ambient * h;
		vec4 diffuse = gl_LightSource[0].diffuse * intensity;
		vec4 specular = gl_LightSource[0].specular * pow(glare, 64.0);
		gl_FragColor = (diffuse + specular) * color + ambient;
	}
''')
simple_bump_prog = shader.ShaderProgram(vert_shader, simple_bump_frag_shader)

crinkle_frag_shader = shader.FragmentShader('crinkle_bump', shader_noise_glsl + '''
	/* normal mapper using turbulent noise */

	uniform float height;
	uniform int detail;
	varying vec3 position;
	varying vec3 normal;
	varying vec3 lightvec;
	varying vec4 color;

	void main(void) {
		const float offset = 0.05; // offset for computing surface derivative

		/* Calculate the normal resulting from the bump surface */
		float h = fbmnoise(position, detail);
		float f0 = h * height;
		float fx = fbmnoise(position + vec3(offset, 0.0, 0.0), detail) * height;
		float fy = fbmnoise(position + vec3(0.0, offset, 0.0), detail) * height;
		float fz = fbmnoise(position + vec3(0.0, 0.0, offset), detail) * height;
		vec3 df = vec3((fx - f0) / offset, (fy - f0) / offset, (fz - f0) / offset);
		vec3 N = normalize(normal - df);

		/* Calculate the lighting */
		float intensity = max(0.0, dot(N, normalize(lightvec)));
		float glare = max(0.0, dot(N, normalize(gl_LightSource[0].halfVector.xyz)));
		h = (h + 1.0) * 0.5;
		vec4 ambient = gl_LightSource[0].ambient * h;
		vec4 diffuse = gl_LightSource[0].diffuse * intensity;
		vec4 specular = gl_LightSource[0].specular * pow(glare, 64.0);
		gl_FragColor = (diffuse + specular) * color + ambient;
	}
''')
crinkle_prog = shader.ShaderProgram(vert_shader, crinkle_frag_shader)

emboss_frag_shader = shader.FragmentShader('emboss_bump', shader_noise_glsl + '''
	/* normal mapper using embossed turbulent noise */

	uniform float height;
	uniform int detail;
	varying vec3 position;
	varying vec3 normal;
	varying vec3 lightvec;
	varying vec4 color;

	float stripes(float x) {
		float t = .5 + .5 * sin(twopi * x);
  		return t * t - .5;
	}

	void main(void) {
		const float offset = 0.001; // offset for computing surface derivative

		/* Calculate the normal resulting from the bump surface */
		vec3 p = position * 1.6;
		vec3 np = position * 0.2;
		float h = stripes(p.x + 2.0 * fbmturbulence(np, detail));
		float f0 = h * height;
		float fx = stripes(p.x +  2.0 * fbmturbulence(np + vec3(offset, 0.0, 0.0), detail)) * height;
		float fy = stripes(p.x + 2.0 * fbmturbulence(np + vec3(0.0, offset, 0.0), detail)) * height;
		float fz = stripes(p.x + 2.0 * fbmturbulence(np + vec3(0.0, 0.0, offset), detail)) * height;
		vec3 df = vec3((fx - f0) / offset, (fy - f0) / offset, (fz - f0) / offset);
		vec3 N = normalize(normal - df);

		/* Calculate the lighting */
		float intensity = max(0.0, dot(N, normalize(lightvec)));
		float glare = max(0.0, dot(N, normalize(gl_LightSource[0].halfVector.xyz)));
		h = (h + 1.0) * 0.01;
		vec4 ambient = gl_LightSource[0].ambient + h;
		vec4 diffuse = gl_LightSource[0].diffuse * intensity;
		vec4 specular = gl_LightSource[0].specular * pow(glare, 64.0);
		gl_FragColor = (diffuse + specular) * color + ambient;
	}
''')
emboss_prog = shader.ShaderProgram(vert_shader, emboss_frag_shader)


if __name__ == '__main__':
	import sys
	global xrot, yrot, d
	win = pyglet.window.Window(width=640, height=640, resizable=True, visible=False,
		config=pyglet.gl.Config(sample_buffers=1, samples=4, double_buffer=True, depth_size=24))

	glEnable(GL_LIGHTING)
	glEnable(GL_LIGHT0)
	fourfv = ctypes.c_float * 4
	glLightfv(GL_LIGHT0, GL_POSITION, fourfv(-250, 250, 500, 1.0))
	glLightfv(GL_LIGHT0, GL_AMBIENT, fourfv(0.1, 0.1, 0.1, 1.0))
	glLightfv(GL_LIGHT0, GL_DIFFUSE, fourfv(0.9, 0.9, 0.9, 1.0))
	glLightfv(GL_LIGHT0, GL_SPECULAR, fourfv(1.5, 1.5, 1.5, 1.0))
	glEnable(GL_COLOR_MATERIAL)
	glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE)
	glMaterialfv(GL_FRONT, GL_SPECULAR, fourfv(1.0, 1.0, 1.0, 1.0))
	glMateriali(GL_FRONT, GL_SHININESS, 128)

	noisetex = ShaderNoiseTexture()
	noisetex.load()
	noisetex.enable()

	quadratic = gluNewQuadric()
	gluQuadricNormals(quadratic, GLU_SMOOTH)
	glEnable(GL_CULL_FACE)
	yrot = spin = 0.0
	xrot = 90.0

	# Setup shader state
	simple_bump_prog.install()
	simple_bump_prog.uset1F('scale', 1.0)
	simple_bump_prog.uset1F('height', 0.045)

	crinkle_prog.install()
	crinkle_prog.uset1F('scale', 1.0)
	crinkle_prog.uset1F('height', -0.02)
	crinkle_prog.uset1I('detail', 4)

	emboss_prog.install()
	emboss_prog.uset1F('scale', 1.0)
	emboss_prog.uset1F('height', 0.003)
	emboss_prog.uset1I('detail', 4)

	def on_resize(width, height):
		glViewport(0, 0, width, height)
		glMatrixMode(GL_PROJECTION)
		glLoadIdentity()
		glOrtho(-2.5, 2.5, -2.5, 2.5, 1, -1)
		glMatrixMode(GL_MODELVIEW)
		glLoadIdentity()
	win.on_resize = on_resize

	@win.event
	def on_draw():
		global xrot, yrot
		win.clear()
		glLoadIdentity()
		glTranslatef(-1.25, 1.25, 0)
		glRotatef(xrot, 1.0, 0.0, 0.0)
		glRotatef(yrot, 0.0, 1.0, 0.0)
		glRotatef(spin, 0.0, 0.0, 1.0)
		simple_bump_prog.install()
		glColor3f(0.9, 0.3, 0.3)
		gluSphere(quadratic, 1.0, 20, 20)
		simple_bump_prog.uninstall()

		glLoadIdentity()
		glTranslatef(1.25, 1.25, 0)
		glRotatef(xrot, 1.0, 0.0, 0.0)
		glRotatef(yrot, 0.0, 1.0, 0.0)
		glRotatef(spin, 0.0, 0.0, 1.0)
		crinkle_prog.install()
		glColor3f(0.3, 0.7, 0.3)
		gluSphere(quadratic, 1.0, 20, 20)
		crinkle_prog.uninstall()

		glLoadIdentity()
		glTranslatef(0, -1, 0)
		glRotatef(xrot, 1.0, 0.0, 0.0)
		glRotatef(yrot, 0.0, 1.0, 0.0)
		glRotatef(spin, 0.0, 0.0, 1.0)
		emboss_prog.install()
		glColor3f(0.3, 0.3, 0.8)
		gluSphere(quadratic, 1.0, 20, 20)
		emboss_prog.uninstall()

	def update(dt):
		global spin
		spin += dt * 10.0

	pyglet.clock.schedule_interval(update, 1.0/30.0)
	
	win.set_visible()
	pyglet.app.run()
