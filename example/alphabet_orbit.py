import time
import OpenGL
OpenGL.ERROR_LOGGING = False
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GL.shaders import compileShader,compileProgram
from ctypes import c_void_p
import numpy as np

import poly_ops

COORD_BYTE_SIZE = 4
POINT_BYTE_SIZE = COORD_BYTE_SIZE * 2

NP_COORD = np.int32

V_SOLID_SHADER_SRC = b"""#version 330 core
layout (location = 0) in vec2 pos;
uniform mat3 world_mat;
void main() {
    vec3 c = world_mat * vec3(pos / 10.0f,1.0f);
    gl_Position = vec4(c.x-1,1-c.y,0.0f,1.0f);
}"""

F_SOLID_SHADER_SRC = b"""#version 330 core
out vec4 FragColor;
uniform vec4 color;
void main() {
    FragColor = color;
}"""

def read_loops(filename):
    with open(filename) as fin:
        data = fin.read()

    r = []
    for shape in data.split('#'):
        loops = []
        for loop in shape.split('='):
            loop = loop.strip()
            if not loop: continue
            loop = np.fromstring(loop,dtype=NP_COORD,sep=' ')
            if not len(loop): continue
            if len(loop) % 2:
                raise RuntimeError('input in "{}" has odd number of coordinates'.format(filename))
            loops.append(np.reshape(loop,(-1,2)))
        if loops: r.append(loops)

    if not r:
        raise RuntimeError('no data in "{}"'.format(filename))

    return r

def rand_orbit(count):
    assert count > 0

    rgen = np.random.default_rng(1)
    theta = rgen.uniform(-np.pi,np.pi,count)
    a = np.ndarray((count,3))
    a[:,0] = np.cos(theta)
    a[:,1] = np.sin(theta)
    a[:,2] = 0
    out = np.array((0.0,0.0,1.0))
    right = np.cross(a,out)
    theta = rgen.normal(0,0.8,count)
    b = (right * np.reshape(np.cos(theta),(-1,1))) + (out * np.reshape(np.sin(theta),(-1,1)))

    radius = np.reshape(rgen.uniform(1000.0,3000.0,count),(-1,1))
    return zip(a[:,:2] * radius,b[:,:2] * radius)

class Satellite:
    def __init__(self,loops,orbit):
        self.loops = loops
        self.orbit = orbit

    def get_coords(self,delta,offset):
        return ((loop +
            (self.orbit[0]*np.sin(delta) +
                self.orbit[1]*np.cos(delta) + offset).astype(NP_COORD))
            for loop in self.loops)

class Scene:
    def __init__(self,alpha_path,center_path):
        self.shapes = []
        data = read_loops(alpha_path)
        for shape,orbit in zip(data,rand_orbit(len(data))):
            self.shapes.append(Satellite(shape,orbit))

        data = read_loops(center_path)
        if len(data) != 1:
            raise RuntimeError('multiple shapes not allowed for "{}"'.format(center_path))
        self.center_shape = data[0]

        self.start_time = time.monotonic()

    def get_current_loops(self,offset):
        offset = np.array(offset,dtype=NP_COORD)

        delta = time.monotonic() - self.start_time

        input_loops = []
        for shape in self.shapes:
            input_loops.extend(shape.get_coords(delta,offset))

        input_loops.extend(loop + offset for loop in self.center_shape)

        return poly_ops.normalize_flat(input_loops)

class Program:
    def __init__(self):
        self.prog = None
        for u in self.uniforms:
            self.__dict__[u] = None
        self.setup_gl()

    def setup_gl(self):
        self.prog = compileProgram(
            compileShader(self.v_source,GL_VERTEX_SHADER),
            compileShader(self.f_source,GL_FRAGMENT_SHADER))
        self.use()
        for u in self.uniforms:
            self.__dict__[u] = glGetUniformLocation(self.prog,u)

    def free(self):
        if self.prog: glDeleteProgram(self.prog)

    def use(self):
        glUseProgram(self.prog)

class SolidProgram(Program):
    v_source = V_SOLID_SHADER_SRC
    f_source = F_SOLID_SHADER_SRC

    uniforms = ['color','world_mat']

    def set_color(self,val):
        glUniform4fv(self.color,1,val)

    def set_world_mat(self,val):
        glUniformMatrix3fv(self.world_mat,1,True,val)

class GLState:
    def __init__(self,scene):
        self.scene = scene
        self.varray = glGenVertexArrays(1)
        self.buffer = glGenBuffers(1)
        self.buffer_size = 0
        self.data_size = 0
        self.program = SolidProgram()
        self.program.set_color([0,0,0,1])

        glBindVertexArray(self.varray)
        glBindBuffer(GL_ARRAY_BUFFER,self.buffer)

        glVertexAttribPointer(0,2,GL_INT,GL_FALSE,8,c_void_p(0))
        glEnableVertexAttribArray(0)

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA)

        glClearColor(1,1,1,1)

    def free(self):
        glDeleteBuffers(1,self.buffer)
        glDeleteVertexArrays(1,self.varray)
        self.program.free()

    def draw(self):
        screen_w = glutGet(GLUT_WINDOW_WIDTH)
        screen_h = glutGet(GLUT_WINDOW_HEIGHT)
        self.program.set_world_mat([
            2/screen_w,0,0,
            0,2/screen_h,0,
            0,0,1])

        glClear(GL_COLOR_BUFFER_BIT)

        for loop in self.scene.get_current_loops((3000,3000)):
            if self.buffer_size >= len(loop):
                glBufferSubData(
                    GL_ARRAY_BUFFER,
                    0,
                    loop.nbytes,
                    loop)
            else:
                glBufferData(GL_ARRAY_BUFFER,loop.nbytes,loop,GL_DYNAMIC_DRAW)
                self.buffer_size = len(loop)

            glDrawArrays(GL_LINE_LOOP,0,len(loop))

        glutSwapBuffers()

def main():
    scene = Scene('alphabet.txt','center.txt')

    glutInit()
    glutInitDisplayMode(GLUT_RGBA|GLUT_DOUBLE)
    glutInitWindowSize(600,600)

    win = glutCreateWindow("Alphabet Orbit")

    gstate = GLState(scene)

    glutDisplayFunc(gstate.draw)
    glutIdleFunc(gstate.draw)
    glutMainLoop()

    gstate.free()

if __name__ == '__main__':
    main()
