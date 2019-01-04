
import OpenGL.GLUT as glut
import OpenGL.GL as gl 
import OpenGL.GLU as glu
import sys
import numpy as np
import ctypes
import math
import glm

theta = 0

vertexShaderCode = """
    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;
    attribute vec3 position;
    attribute vec4 color;
    varying vec4 v_color;
    void main()
    {
        gl_Position = projection * view * model * vec4(position, 1.0);
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
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
    offset = ctypes.c_void_p(0)
    gl.glDrawElements(gl.GL_TRIANGLES, 36, gl.GL_UNSIGNED_INT, offset)
    glut.glutSwapBuffers()

def idle():
    global theta
    loc = gl.glGetUniformLocation(program, "model")
    model = glm.mat4(1.0)
    model = glm.rotate(model, theta, (1.0, 0.0, 0.0))
    model = glm.rotate(model, theta, (0.0, 1.0, 0.0))
    gl.glUniformMatrix4fv(loc, 1, gl.GL_FALSE, glm.value_ptr(model))
    theta += 0.01
    if theta > 2 * math.pi:
        theta -= 2 * math.pi

    draw()

def reshape(width, height):
    gl.glViewport(0, 0, width, height)
    gl.glDepthFunc(gl.GL_LESS)
    gl.glEnable(gl.GL_DEPTH_TEST)
    projection = glm.perspective(45.0, 1.0, 0.5, 100.0)
    loc = gl.glGetUniformLocation(program, "projection")
    gl.glUniformMatrix4fv(loc, 1, gl.GL_FALSE, glm.value_ptr(projection))

def keyboard(key, x, y):
    if key == b'\x1b':
        sys.exit()

def makePerspective(near, far, width, height):
    P = np.eye(4, dtype=np.float32)
    P[0][0] = 2 * near / width
    P[1][1] = 2 * near / height
    P[2][2] = -1.0 * (far + near) / (far - near)
    P[3][2] = -2.0 * far * near / (far - near) 
    P[2][3] = -1.0
    return P

def LoadShaders():
    program = gl.glCreateProgram()
    vertex = gl.glCreateShader(gl.GL_VERTEX_SHADER)
    fragment = gl.glCreateShader(gl.GL_FRAGMENT_SHADER)

    gl.glShaderSource(vertex, vertexShaderCode)
    gl.glShaderSource(fragment, fragmentShaderCode)

    gl.glCompileShader(vertex)
    if not gl.glGetShaderiv(vertex, gl.GL_COMPILE_STATUS):
        error = gl.glGetShaderInfoLog(vertex).decode()
        print (error)
        raise RuntimeError("Vertex Shader Compile error")

    gl.glCompileShader(fragment)
    if not gl.glGetShaderiv(fragment, gl.GL_COMPILE_STATUS):
        error = gl.glGetShaderInfoLog(fragment).decode()
        print (error)
        raise RuntimeError("Frag Shader Compile error")

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

    # Vertices
    V = np.zeros((8,3), dtype=np.float32)
    V[...] = [ 1, 1, 1], [-1, 1, 1], [-1,-1, 1], [ 1,-1, 1],[ 1,-1,-1], [ 1, 1,-1], [-1, 1,-1], [-1,-1,-1]

    vbuffer = gl.glGenBuffers(1)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbuffer)
    gl.glBufferData(gl.GL_ARRAY_BUFFER, V.nbytes, V, gl.GL_DYNAMIC_DRAW)

    stride = V.strides[0]
    offset = ctypes.c_void_p(0)
    loc = gl.glGetAttribLocation(program, "position")
    gl.glEnableVertexAttribArray(loc)
    gl.glVertexAttribPointer(loc, 3, gl.GL_FLOAT, False, stride, offset)

    # Vertice Colours
    C = np.zeros((8,4), dtype=np.float32)
    C[...] = (1,1,1,1),(1,1,0,1),(1,0,1,1),(1,0,0,1),(0,1,1,1),(0,1,0,1),(0,0,0,1),(0,1,1,1)
    cbuffer = gl.glGenBuffers(1)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, cbuffer)
    gl.glBufferData(gl.GL_ARRAY_BUFFER, C.nbytes, C, gl.GL_DYNAMIC_DRAW)
    stride = C.strides[0]
    offset = ctypes.c_void_p(0)
    loc = gl.glGetAttribLocation(program, "color")
    gl.glEnableVertexAttribArray(loc)
    gl.glVertexAttribPointer(loc, 4, gl.GL_FLOAT, False, stride, offset)

    # Indexes
    I = np.array([0,1,2, 0,2,3, 0,3,4, 0,4,5, 0,5,6, 0,6,1, 1,6,7, 1,7,2, 7,4,3, 7,3,2, 4,7,6, 4,6,5], dtype=np.uint32)
    ibuffer = gl.glGenBuffers(1)
    gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, ibuffer)
    gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, I.nbytes, I, gl.GL_DYNAMIC_DRAW)


    # MVP Matrices
    projection = glm.perspective(45.0, 1.0, 0.5, 100.0)
    print projection
    loc = gl.glGetUniformLocation(program, "projection")
    gl.glUniformMatrix4fv(loc, 1, gl.GL_FALSE, glm.value_ptr(projection))

    loc = gl.glGetUniformLocation(program, "view")
    view = np.eye(4, dtype=np.float32)
    view[3][2] = -5.0
    gl.glUniformMatrix4fv(loc, 1, gl.GL_FALSE, view)

    loc = gl.glGetUniformLocation(program, "model")
    model = glm.mat4(1.0)
    gl.glUniformMatrix4fv(loc, 1, gl.GL_FALSE, glm.value_ptr(model))



glut.glutInit(sys.argv)
glut.glutInitDisplayMode(glut.GLUT_DOUBLE | glut.GLUT_RGBA)
glut.glutInitWindowSize(250,250)
glut.glutInitWindowPosition(100,100)
glut.glutCreateWindow("OGL Program")
program = LoadShaders()
LoadData(program)
print gl.glGetString(gl.GL_VERSION)
glut.glutDisplayFunc(draw)
glut.glutKeyboardFunc(keyboard)
glut.glutReshapeFunc(reshape)
glut.glutIdleFunc(idle)
glut.glutMainLoop()