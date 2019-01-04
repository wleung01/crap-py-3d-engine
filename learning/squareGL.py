"""
This learning example animates a rotating, shrinking & growing square by using
4 Vertices, and a fragment shader that interpolates colours across the 4 vertices.
Scaling and Rotation is done in the Vertex Shader.

Thanks goes to  Nicolas P. Rougier as I was able to learn this much from his book 
which is available online for learning:
http://www.labri.fr/perso/nrougier/python-opengl/
"""
import OpenGL.GLUT as glut
import OpenGL.GL as gl 
import OpenGL.GLU as glu
import sys
import numpy as np
import ctypes
import math

vertexShaderCode = """
    uniform float scale;
    uniform float theta;
    attribute vec2 position;
    attribute vec4 color;
    varying vec4 v_color;
    void main()
    {
        float x2 = cos(theta)*position.x - sin(theta)*position.y;
        float y2 = sin(theta)*position.x + cos(theta)*position.y;
        gl_Position = vec4(x2 * scale, y2 * scale, 0.0, 1.0);
        v_color = color;
    }
"""

fragmentShaderCode = """
    varying vec4 v_color;
    void main()
    {
        gl_FragColor = v_color;
    }
"""

def draw():
    gl.glClear(gl.GL_COLOR_BUFFER_BIT)
    gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, 4)
    glut.glutSwapBuffers()

scale = 1.0
scale_inc = -0.01

theta = 0.0
theta_inc = 0.05

def idle():
    global scale
    global scale_inc
    global theta
    global theta_inc
    loc = gl.glGetUniformLocation(program, "scale")
    gl.glUniform1f(loc, scale)
    loc = gl.glGetUniformLocation(program, "theta")
    gl.glUniform1f(loc, theta)
    scale += scale_inc
    if scale < -1.0 or scale > 1.0:
        scale_inc *= -1

    theta = theta + theta_inc
    if (theta > 2 * math.pi):
        theta -= 2 * math.pi

    draw()

def reshape(width, height):
    gl.glViewport(0, 0, width, height)

def keyboard(key, x, y):
    if key == b'\x1b':
        sys.exit()

def CompileShader(shader, szType):
    gl.glCompileShader(shader)
    if not gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS):
        print(gl.glGetShaderInfoLog(shader).decode())
        raise RuntimeError("%s Shader Compile error" % szType)


def LoadShaders():
    program = gl.glCreateProgram()
    vertex = gl.glCreateShader(gl.GL_VERTEX_SHADER)
    fragment = gl.glCreateShader(gl.GL_FRAGMENT_SHADER)

    gl.glShaderSource(vertex, vertexShaderCode)
    gl.glShaderSource(fragment, fragmentShaderCode)

    CompileShader(vertex, "Vertex")
    CompileShader(fragment, "Fragment")

    gl.glAttachShader(program, vertex)
    gl.glAttachShader(program, fragment)
    gl.glLinkProgram(program)
    if not gl.glGetProgramiv(program, gl.GL_LINK_STATUS):
        print (gl.glGetProgramInfoLog(program))
        raise RuntimeError("Link fail")

    gl.glDetachShader(program, vertex)
    gl.glDetachShader(program, fragment)

    gl.glUseProgram(program)

    return program

def LoadData(program):
    LoadVertices(program)

    LoadVertexColours(program)

def LoadVertices(program):
    v = np.zeros((4,2), dtype=np.float32)
    v[...] = (-1,+1), (1, 1), (-1, -1), (1, -1)

    buffer = gl.glGenBuffers(1)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, buffer)
    gl.glBufferData(gl.GL_ARRAY_BUFFER, v.nbytes, v, gl.GL_DYNAMIC_DRAW)

    stride = v.strides[0]
    offset = ctypes.c_void_p(0)
    loc = gl.glGetAttribLocation(program, "position")
    gl.glEnableVertexAttribArray(loc)
    gl.glVertexAttribPointer(loc, 2, gl.GL_FLOAT, False, stride, offset)

def LoadVertexColours(program):
    c = np.zeros((4,4), dtype=np.float32)
    c[...] = (1,1,0,1), (1,0,0,1), (0,0,1,1), (0,1,0,1)
    c_buffer = gl.glGenBuffers(1)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, c_buffer)
    gl.glBufferData(gl.GL_ARRAY_BUFFER, c.nbytes, c, gl.GL_DYNAMIC_DRAW)
    stride = c.strides[0]
    offset = ctypes.c_void_p(0)
    loc = gl.glGetAttribLocation(program, "color")
    gl.glEnableVertexAttribArray(loc)
    gl.glVertexAttribPointer(loc, 4, gl.GL_FLOAT, False, stride, offset)


glut.glutInit(sys.argv)
glut.glutInitDisplayMode(glut.GLUT_DOUBLE | glut.GLUT_RGBA)
glut.glutInitWindowSize(250,250)
glut.glutInitWindowPosition(100,100)
glut.glutCreateWindow("Rotating Shrinking Square Program")
program = LoadShaders()
LoadData(program)
print gl.glGetString(gl.GL_VERSION)
glut.glutDisplayFunc(draw)
glut.glutKeyboardFunc(keyboard)
glut.glutReshapeFunc(reshape)
glut.glutIdleFunc(idle)
glut.glutMainLoop()