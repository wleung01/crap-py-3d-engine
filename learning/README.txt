The following steps are needed to get my learning examples to work on MS Windows 10.

- Install PyOpenGL from the .whl file in 'libraries'. Using pip install will not work on Windows for some reason.
- pip install numpy 
- pip install pyglm

The following order of examples show the progression I made as I learned things. I suppose it should be a ranking of complexity
Order of learning examples:
- squareGL.py
    - Compiling, Linking shaders and using program
    - Sending Buffer Data, and VertexAttribute Data to GPU
    - Basic Shader use
    - Basic GLUT Use
- cubeGL.py
    - Using Index Buffers to reduce Vertex Buffer Size
    - Using Model, View & Projection matrices to position camera & target.
    - Enabling of Depth Testing, and the importance of clearing GL_DEPTH_BUFFER_BIT