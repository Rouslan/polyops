import itertools
import unittest
import numpy as np
import numpy.linalg as alg

import polyops


def index_of(arr,val,*,delta=0):
    if delta:
        mask = np.isclose(arr,val,atol=delta)
    else:
        mask = arr == val
    i, = np.where(np.all(mask,1))
    if len(i) == 0: return -1
    return i[0]


def vangle(a,b):
    return np.arccos(np.vecdot(a,b)/(alg.vector_norm(a,axis=-1)*alg.vector_norm(b,axis=-1)))

def perp_vector(p1,p2,magnitude):
    perp = np.stack((p2[...,1] - p1[...,1],p1[...,0] - p2[...,0]),axis = -1)
    return perp * (magnitude/alg.vector_norm(perp,axis=-1))[...,None]


class Tests(unittest.TestCase):
    def assertNumpyEqual(self,a,b,*,delta=0) -> None:
        if delta:
            if not np.allclose(a,b,atol=delta):
                raise self.failureException(f'{a} != {b} (tolerance of {delta})')
        else:
            if not np.array_equal(a,b):
                raise self.failureException(f'{a} != {b}')

    def test_decomposition(self):
        loops = [[
            [280,220],[249,215],[221,200],[199,178],[184,150],[180,119],
            [184,89],[199,61],[221,39],[249,24],[280,19],[310,24],[338,39],
            [360,61],[375,89],[380,119],[375,150],[360,178],[338,200],[310,215]
        ],[
            [120,380],[89,375],[61,360],[39,338],[24,310],[19,280],[24,249],
            [39,221],[61,199],[89,184],[120,179],[150,184],[178,199],[200,221],
            [215,249],[220,280],[215,310],[200,338],[178,360],[150,375]
        ],[
            [240,300],[209,295],[181,280],[159,258],[144,230],[140,200],
            [144,169],[159,141],[181,119],[209,104],[240,99],[270,104],
            [298,119],[320,141],[335,169],[340,200],[335,230],[320,258],
            [298,280],[270,295]
        ],[
            [200,340],[169,335],[141,320],[119,298],[104,270],[99,240],
            [104,209],[119,181],[141,159],[169,144],[200,139],[230,144],
            [258,159],[280,181],[295,209],[300,240],[295,270],[280,298],
            [258,320],[230,335]
        ],[
            [200,260],[169,255],[141,240],[119,218],[104,190],[99,160],
            [104,129],[119,101],[141,79],[169,64],[200,59],[230,64],[258,79],
            [280,101],[295,129],[300,160],[295,190],[280,218],[258,240],
            [230,255]
        ]]

        for p in itertools.permutations(loops):
            # This shape should decompose into exactly 10 shapes and 0 holes
            out = polyops.normalize(p,tree_out=True)
            self.assertEqual(len(out),10)
            self.assertTrue(all(len(loop[1]) == 0 for loop in out))

    def test_nesting(self):
        loops = [[
            [ 20, 20],[157, 43],[256, 17],[338, 30],[356, 89],[363,189],
            [130,204],[ 14,185],[ 36, 95]
        ],[
            [ 63,155],[ 58,103],[ 95, 66],[150, 74],[176,106],[178,146],
            [150,176],[111,179]
        ],[
            [265,165],[207,165],[196,115],[212, 62],[298, 46],[336,101],[332,145]
        ],[
            [118,157],[ 81,136],[ 80,103],[120, 89],[156,112],[149,144]
        ],[
            [238,149],[218, 74],[287, 58],[320,123],[287,142]
        ],[
            [107,130],[101,104],[138,118],[126,143]
        ],[
            [188, 87],[157, 57],[197, 53]
        ]]

        out = list(polyops.normalize(loops,tree_out=True))
        self.assertEqual(len(out), 1)
        self.assertEqual(len(out[0].loop), 9)
        self.assertEqual(len(out[0].children), 3)
        out = sorted(out[0][1], key=(lambda x: len(x[0])))

        self.assertEqual(len(out[0].loop), 3)
        self.assertEqual(len(out[0].children), 0)

        self.assertEqual(len(out[1].loop), 7)
        self.assertEqual(len(out[1].children), 1)
        self.assertEqual(len(out[1].children[0].loop), 5)
        self.assertEqual(len(out[1].children[0][1]), 0)

        self.assertEqual(len(out[2].loop), 8)
        self.assertEqual(len(out[2].children), 1)
        self.assertEqual(len(out[2].children[0].loop), 6)
        self.assertEqual(len(out[2].children[0].children), 1)
        self.assertEqual(len(out[2].children[0].children[0].loop), 4)
        self.assertEqual(len(out[2].children[0].children[0].children), 0)

    def test_basic_offset(self):
        box = [[[0,0],[1000,0],[1000,1000],[0,1000]]]
        result = polyops.offset(box,50,1000000,track_points=True)
        self.assertEqual(len(result),1)

        box2 = result[0]
        self.assertEqual(len(box2.loop),8)

        # start at point [-50,0]
        i=index_of(box2.loop,[-50,0])
        self.assertGreaterEqual(i,0)

        self.assertNumpyEqual(box2.originals[i],[0])
        i = (i+1)%8
        self.assertNumpyEqual(box2.loop[i],[0,-50])
        self.assertNumpyEqual(box2.originals[i],[0])

        i = (i+1)%8
        self.assertNumpyEqual(box2.loop[i],[1000,-50])
        self.assertNumpyEqual(box2.originals[i],[1])
        i = (i+1)%8
        self.assertNumpyEqual(box2.loop[i],[1050,0])
        self.assertNumpyEqual(box2.originals[i],[1])

        i = (i+1)%8
        self.assertNumpyEqual(box2.loop[i],[1050,1000])
        self.assertNumpyEqual(box2.originals[i],[2])
        i = (i+1)%8
        self.assertNumpyEqual(box2.loop[i],[1000,1050])
        self.assertNumpyEqual(box2.originals[i],[2])

        i = (i+1)%8
        self.assertNumpyEqual(box2.loop[i],[0,1050])
        self.assertNumpyEqual(box2.originals[i],[3])
        i = (i+1)%8
        self.assertNumpyEqual(box2.loop[i],[-50,1000])
        self.assertNumpyEqual(box2.originals[i],[3])

    def test_compound_offset(self):
        boxes = [
            [[0,0],[1000,0],[1000,1000],[0,1000]],
            [[2000,0],[3000,0],[3000,1000],[2000,1000]],
            [[4000,0],[5000,0],[5000,1000],[4000,1000]]]
        result = polyops.offset(boxes,50,1000000,track_points=True)
        self.assertEqual(len(result),3)

        for j in range(3):
            x_offset = j*2000
            index_offset = j*4
            box2 = result[j]
            self.assertEqual(len(box2.loop),8)

            # start at point [-50+x_offset,0]
            i=index_of(box2.loop,[-50+x_offset,0])
            self.assertGreaterEqual(i,0)

            self.assertNumpyEqual(box2.originals[i],[index_offset])
            i = (i+1)%8
            self.assertNumpyEqual(box2.loop[i],[x_offset,-50])
            self.assertNumpyEqual(box2.originals[i],[index_offset])

            i = (i+1)%8
            self.assertNumpyEqual(box2.loop[i],[1000+x_offset,-50])
            self.assertNumpyEqual(box2.originals[i],[1+index_offset])
            i = (i+1)%8
            self.assertNumpyEqual(box2.loop[i],[1050+x_offset,0])
            self.assertNumpyEqual(box2.originals[i],[1+index_offset])

            i = (i+1)%8
            self.assertNumpyEqual(box2.loop[i],[1050+x_offset,1000])
            self.assertNumpyEqual(box2.originals[i],[2+index_offset])
            i = (i+1)%8
            self.assertNumpyEqual(box2.loop[i],[1000+x_offset,1050])
            self.assertNumpyEqual(box2.originals[i],[2+index_offset])

            i = (i+1)%8
            self.assertNumpyEqual(box2.loop[i],[x_offset,1050])
            self.assertNumpyEqual(box2.originals[i],[3+index_offset])
            i = (i+1)%8
            self.assertNumpyEqual(box2.loop[i],[-50+x_offset,1000])
            self.assertNumpyEqual(box2.originals[i],[3+index_offset])

    def test_offset_curves(self):
        def variant(clipper=None):
            self.tracked_offset_check([[0,0],[1000,0],[1000,1000],[0,1000]],50,10,clipper)
            self.tracked_offset_check([[3225,-3225],[5450,-13525],[16000,-15450],[8025,-1575]],2000,100,clipper)
            self.tracked_offset_check([[20,20],[140,35],[37,142],[20,100]],20,15,clipper)
        
        variant()
        variant(polyops.TrackedClipper())
    
    # this only works for simple convex shapes
    def tracked_offset_check(self,loop,offset,arc_step_size,clipper):
        loop = np.array(loop)
        loop_before = np.roll(loop,1,0)
        loop_after = np.roll(loop,-1,0)
        angles = np.pi - vangle(loop_before - loop,loop_after - loop)

        seg_per_curve = (angles * offset / arc_step_size).astype(int)
        seg_per_curve[seg_per_curve < 1] = 1
        total_points = np.sum(seg_per_curve)+len(loop)

        curve_starts = perp_vector(loop_before,loop,offset) + loop
        curve_ends = perp_vector(loop,loop_after,offset) + loop

        if clipper is None:
            result = polyops.offset([loop],offset,arc_step_size,track_points=True)
        else:
            clipper.add_offset_subject(loop,offset,arc_step_size)
            result = clipper.execute(polyops.BoolOp.union)
        self.assertEqual(len(result),1)
        box2 = result[0]
        self.assertEqual(len(box2.loop),total_points)

        # start at point curve_starts[0]
        i=index_of(box2.loop,curve_starts[0],delta=1)
        self.assertGreaterEqual(i,0)

        prev_i = 0
        for segs,c_start,c_end,k in zip(seg_per_curve,curve_starts,curve_ends,range(len(seg_per_curve))):
            self.assertNumpyEqual(box2.loop[i],c_start,delta=1)
            for j in range(segs+1):
                prev_i = i
                self.assertNumpyEqual(box2.originals[i],[k])
                i = (i+1)%total_points
            self.assertNumpyEqual(box2.loop[prev_i],c_end,delta=1)

        unmap = box2.originals.index_map()
        self.assertNumpyEqual(unmap,np.arange(total_points))


if __name__ == '__main__':
    unittest.main()
