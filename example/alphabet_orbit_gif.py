
import numpy as np
from PIL import Image,ImageDraw

import poly_ops

NP_COORD = np.int32


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

    def get_current_loops(self,delta,offset):
        offset = np.array(offset,dtype=NP_COORD)

        input_loops = []
        for shape in self.shapes:
            input_loops.extend(shape.get_coords(delta,offset))

        input_loops.extend(loop + offset for loop in self.center_shape)

        return poly_ops.normalize_flat(input_loops)


def main():
    scene = Scene('alphabet.txt','center.txt')
    images = []
    for i in range(240):
        img = Image.new('L',(800,800),255)
        images.append(img)

        delta = (np.pi / 120) * i

        draw = ImageDraw.Draw(img)
        for loop in scene.get_current_loops(delta,(4000,4000)):
            draw.polygon(list(np.reshape(loop,(-1,))//10),outline=0)
    
    images[0].save('alphabet_orbit.gif',save_all=True,append_images=images[1:],duration=30,loop=0)

if __name__ == '__main__':
    main()
