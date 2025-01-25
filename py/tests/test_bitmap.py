import sys
import re
import numpy as np
import unittest
import importlib.resources

import polyops
from . import polydraw
from . import test_data

DIFF_FAIL_SQUARE_SIZE = 5
DUMP_FAILURE_DIFF = False
OFFSET_ARC_STEP_SIZE = 3

# Technically, comments can appear anywhere, including inside 'tokens', not just
# around tokens, but we only need to open our own files, so this is good enough.
# The 'P4' at the start is omitted because it is checked separately.
RE_PBM_HEADER = re.compile(
    br"(?:#.*[\n\r])*\s(?:#.*[\n\r]|\s)*"
    + br"(\d+)"
    + br"(?:#.*[\n\r])*\s(?:#.*[\n\r]|\s)*"
    + br"(\d+)"
    + br"(?:#.*[\n\r])*\s")


test_data_files = importlib.resources.files(test_data)

class ImgFormatError(Exception):
    pass

def read_pbm(filename,data):
    if not data.startswith(b'P4'):
        raise ImgFormatError('not a binary PBM file',filename)

    m = RE_PBM_HEADER.match(data,2)
    if not m:
        raise ImgFormatError('invalid PBM file',filename)
    width = int(m[1],base=10)
    height = int(m[2],base=10)
    return np.unpackbits(
        np.frombuffer(data,np.ubyte,offset=m.end()).reshape((height,(width+7)//8)),
        axis=1,
        count=width)

def write_pbm(filename,data):
    pbm_buffer = np.packbits(data,axis=1)
    with open(filename,'xb') as out:
        out.write(f'P4 {data.shape[1]} {data.shape[0]} '.encode('ascii'))
        out.write(pbm_buffer)

def has_square(data):
    if data.shape[0] < DIFF_FAIL_SQUARE_SIZE or data.shape[1] < DIFF_FAIL_SQUARE_SIZE:
        raise Exception('image too small for meaningful comparison')

    inner_w = data.shape[1]-DIFF_FAIL_SQUARE_SIZE+1
    inner_h = data.shape[0]-DIFF_FAIL_SQUARE_SIZE+1
    summed = np.ones((inner_h,inner_w),np.ubyte)

    for y in range(DIFF_FAIL_SQUARE_SIZE):
        for x in range(DIFF_FAIL_SQUARE_SIZE):
            summed &= data[y:y+inner_h,x:x+inner_w]

    return np.any(summed)

class TestCase:
    def __init__(self):
        self.set_data = ([],[])
        self.op_files = [{} for i in range(len(polyops.BoolOp))]

class ParseError(Exception):
    pass

def require_whitespace_or_comment(line,lnum):
    line = line.lstrip(' \t\n\r')
    if line and not line.startswith("#"):
        raise ParseError(lnum)

def parse_tests():
    tests = [TestCase()]
    touched = False
    last_set = None
    partial_line = None

    re_array_assign_start = re.compile(r'[ \t]*:[ \t]*\[')
    re_array_assign_end = re.compile(r'([0-9 +\-]*)(\])[ \t\n\r]*(?:$|#)')
    re_op_assign = re.compile(r'([+-][0-9]+|)[ \t]*:[ \t]*([0-9a-zA-Z]+)[ \t\n\r]*(?:$|#)')

    def end_array_assign(m,lnum):
        nonlocal touched, partial_line
        if not m:
            raise ParseError(lnum)
        partial_line.append(m[1])
        if m[2] is None: return
        tests[-1].set_data[last_set].append(np.fromstring(' '.join(partial_line),np.int_,sep=' ').reshape((-1,2)))
        touched = True
        partial_line = None

    with (test_data_files/'input.txt').open() as ifile:
        for i,line in enumerate(ifile):
            if partial_line is not None:
                end_array_assign(re_array_assign_end.match(line),i)
            else:
                for s in polyops.BoolSet:
                    if line.startswith(s.name):
                        last_set = s
                        m = re_array_assign_start.match(line,len(s.name))
                        if not m:
                            raise ParseError(i)
                        partial_line = []
                        end_array_assign(re_array_assign_end.match(line,m.end()),i)
                        break
                else:
                    for op in polyops.BoolOp:
                        if line.startswith(op.name):
                            m = re_op_assign.match(line,len(op.name))
                            if not m:
                                raise ParseError(i)
                            tests[-1].op_files[op][int(m[1],10) if m[1] else 0] = m[2]
                            touched = True
                            break
                    else:
                        if line.startswith("-"):
                            require_whitespace_or_comment(line.lstrip("-"),i)
                            if touched:
                                tests.append(TestCase())
                                touched = False
                        else:
                            require_whitespace_or_comment(line,i)

    if partial_line is not None:
        raise ParseError(-1)

    if not touched: tests.pop()
    return tests

class BitmapTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = parse_tests()

        for item in cls.data:
            if all(op is None for op in item.op_files):
                print("warning: test entry found with nothing to test against",file=sys.stderr)
                break

        cls.clip = polyops.Clipper()
        cls.rast = polydraw.Rasterizer()

    def _run_op(self,op,nonzero_offset=False):
        for item in self.data:
            for offset,file_id in item.op_files[op].items():
                if (offset == 0) == nonzero_offset: continue

                params = dict(op=op,file=file_id)
                if offset:
                    params['offset'] = offset

                with self.subTest(**params):
                    imgfilename = file_id + '.pbm'
                    target_img = read_pbm(imgfilename,(test_data_files/imgfilename).read_bytes())
                    test_img = np.zeros_like(target_img)

                    if offset:
                        self.clip.add_offsets_subject(item.set_data[polyops.BoolSet.subject],offset,OFFSET_ARC_STEP_SIZE)
                        self.clip.add_offsets_clip(item.set_data[polyops.BoolSet.clip],offset,OFFSET_ARC_STEP_SIZE)
                    else:
                        self.clip.add_loops_subject(item.set_data[polyops.BoolSet.subject])
                        self.clip.add_loops_clip(item.set_data[polyops.BoolSet.clip])

                    self.rast.reset()
                    self.rast.add_loops(self.clip.execute(op))
                    for x1,x2,y,wind in self.rast.scan_lines(test_img.shape[1],test_img.shape[0]):
                        if wind > 0:
                            test_img[y,x1:x2+1] = 1

                    diff = test_img != target_img
                    if has_square(diff):
                        if DUMP_FAILURE_DIFF:
                            write_pbm(file_id + "_mine.pbm",test_img)
                            write_pbm(file_id + "_diff.pbm",diff)
                        self.fail('output does not match file')

    def test_union(self):
        self._run_op(polyops.BoolOp.union)

    def test_intersection(self):
        self._run_op(polyops.BoolOp.intersection)

    def test_xor(self):
        self._run_op(polyops.BoolOp.xor)

    def test_difference(self):
        self._run_op(polyops.BoolOp.difference)

    def test_normalize(self):
        self._run_op(polyops.BoolOp.normalize)
    
    def test_offset(self):
        for op in polyops.BoolOp:
            self._run_op(op,True)

if __name__ == '__main__':
    unittest.main()
