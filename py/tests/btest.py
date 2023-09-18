import sys
import re
import numpy as np
import os.path

import poly_ops
import polydraw

DIFF_FAIL_SQUARE_SIZE = 5
DUMP_FAILURE_DIFF = False

# Technically, comments can appear anywhere, including inside 'tokens', not just
# around tokens, but we only need to open our own files, so this is good enough.
# The 'P4' at the start is omitted because it is checked separately.
RE_PBM_HEADER = re.compile(b"""
    (?:#.*[\n\r])*\\s(?:#.*[\n\r]|\\s)*
    (\\d+)
    (?:#.*[\n\r])*\\s(?:#.*[\n\r]|\\s)*
    (\\d+)
    (?:#.*[\n\r])*\\s""",re.VERBOSE)


class ImgFormatError(Exception):
    pass

def read_pbm(filename):
    with open(filename,'rb') as f_in:
        data = f_in.read()
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
        self.op_files = [None] * len(poly_ops.BoolOp)

class ParseError(Exception):
    pass

def require_whitespace_or_comment(line,lnum):
    line = line.lstrip(' \t\n\r')
    if line and not line.startswith("#"):
        raise ParseError(lnum)

def parse_tests(ipath):
    tests = [TestCase()]
    touched = False
    last_set = None
    partial_line = None

    re_array_assign_start = re.compile(r'[ \t]*:[ \t]*\[')
    re_array_assign_end = re.compile(r'([0-9 +\-]*)(\])[ \t\n\r]*(?:$|#)')
    re_op_assign = re.compile(r'[ \t]*:[ \t]*([0-9a-zA-Z]+)[ \t\n\r]*(?:$|#)')

    def end_array_assign(m,lnum):
        nonlocal touched, partial_line
        if not m:
            raise ParseError(lnum)
        partial_line.append(m[1])
        if m[2] is None: return
        tests[-1].set_data[last_set].append(np.fromstring(' '.join(partial_line),np.int_,sep=' ').reshape((-1,2)))
        touched = True
        partial_line = None

    with open(ipath) as ifile:
        for i,line in enumerate(ifile):
            if partial_line is not None:
                end_array_assign(re_array_assign_end.match(line),i)
            else:
                for s in poly_ops.BoolSet:
                    if line.startswith(s.name):
                        last_set = s
                        m = re_array_assign_start.match(line,len(s.name))
                        if not m:
                            raise ParseError(i)
                        partial_line = []
                        end_array_assign(re_array_assign_end.match(line,m.end()),i)
                        break
                else:
                    for op in poly_ops.BoolOp:
                        if line.startswith(op.name):
                            m = re_op_assign.match(line,len(op.name))
                            if not m:
                                raise ParseError(i)
                            tests[-1].op_files[op] = m[1]
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

def test_op(clip,rast,test,op,file_id,folder):
    target_img = read_pbm(os.path.join(folder,file_id + ".pbm"))
    test_img = np.zeros_like(target_img)

    clip.add_loops_subject(test.set_data[poly_ops.BoolSet.subject])
    clip.add_loops_clip(test.set_data[poly_ops.BoolSet.clip])

    rast.reset()
    rast.add_loops(clip.execute_flat(op))
    for x1,x2,y,wind in rast.scan_lines(test_img.shape[1],test_img.shape[0]):
        if wind > 0:
            test_img[y,x1:x2+1] = 1

    diff = test_img != target_img
    if has_square(diff):
        print(f'failure with operation "{op.name}" compared to file "{file_id}"',file=sys.stderr)
        if DUMP_FAILURE_DIFF:
            write_pbm(file_id + "_mine.pbm",test_img)
            write_pbm(file_id + "_diff.pbm",diff)
        return False
    return True

def run(datafile):
    tests = 0
    successes = 0

    cases = parse_tests(datafile)
    
    test_source = os.path.dirname(datafile)

    # make sure bitmap::has_square works
    assert not has_square(read_pbm(os.path.join(test_source,"discont.pbm")))
    assert has_square(read_pbm(os.path.join(test_source,"cont.pbm")))

    clip = poly_ops.Clipper()
    rast = polydraw.Rasterizer()

    for case in cases:
        has_one = False
        for op in poly_ops.BoolOp:
            if case.op_files[op] is not None:
                has_one = True
                successes += test_op(clip,rast,case,op,case.op_files[op],test_source)
                tests += 1

        if not has_one:
            print("warning: test entry found with nothing to test against",file=sys.stderr)
            continue;

    print(f"passed tests: {successes} out of {tests}",file=sys.stderr)

    return tests != successes

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("exactly one argument is required",file=sys.stderr)
        exit(2)
    exit(run(sys.argv[1]))