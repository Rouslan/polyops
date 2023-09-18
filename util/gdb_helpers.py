import subprocess
import json
import os.path
import sys
import gdb
import gdb.printing

MAX_PLOT_SIZE = 400

PLOTTER = os.path.join(os.path.dirname(__file__),'plotter.py')

def cpp_loop_val_to_py(x):
    data = x['data']['_data']
    return {'p': (int(data[0]),int(data[1])), 'next': int(x['next'])}

def vector_length_data(x):
    impl = x['_M_impl']
    start = impl['_M_start']
    end = impl['_M_finish']
    return (int(end-start),start)

def vector_data(x):
    return x['_M_impl']['_M_start']

class Plotter(gdb.Command):
    """Display the lines represented by a vector of poly_ops::detail::loop_point
    instances."""

    def __init__(self):
        super().__init__('plot_lpoints',gdb.COMMAND_DATA,gdb.COMPLETE_SYMBOL)
        self.proc = None

    def invoke(self,argument,from_tty):
        if not argument: raise gdb.GdbError('an argument is required')

        size,data = vector_length_data(gdb.parse_and_eval(argument))

        if size > MAX_PLOT_SIZE: size = MAX_PLOT_SIZE

        points = [cpp_loop_val_to_py(data[i]) for i in range(size)]

        if self.proc is None or self.proc.poll() is not None:
            self.proc = subprocess.Popen([sys.executable,PLOTTER],stdin=subprocess.PIPE,text=True)

        json.dump(points,self.proc.stdin)
        self.proc.stdin.write('\n')
        self.proc.stdin.flush()

class PointPrinter:
    """Print a poly_ops::point_t instance"""
    def __init__(self,val):
        self.val = val

    def to_string(self):
        data = self.val['_data']
        return '{%i,%i}' % (int(data[0]),int(data[1]))

class SweepSetIter:
    def __init__(self,nodes,n,nil):
        self.nodes = nodes
        self.n = n
        self.nil = nil

    def __iter__(self):
        return self

    def __next__(self):
        if int(self.n) == 0:
            raise StopIteration

        node_obj = self.nodes[self.n]
        value = node_obj['value']

        next_n = node_obj['right']
        if next_n != self.nil:
            while True:
                self.n = next_n
                next_n = self.nodes[next_n]['left']
                if next_n == self.nil:
                    break
        else:
            p = node_obj['parent']
            p_obj = self.nodes[p]
            while self.n == p_obj['right']:
                self.n = p
                p = p_obj['parent']
                p_obj = self.nodes[p]

            if self.nodes[self.n]['right'] != p: self.n = p

        return value

def sweep_set_iter(val):
    nodes = vector_data(val['traits_val']['vec'].dereference())
    index_t = val.type.template_argument(1)
    bits = 8 * index_t.sizeof
    if index_t.is_signed:
        bits -= 1
    return SweepSetIter(nodes,nodes[0]['left'],(1 << bits) - 1)

class SweepSetPrinter:
    """Print a poly_ops::detail::sweep_set instance"""
    def __init__(self,val):
        self.val = val

    def __iter__(self):
        return enumerate(sweep_set_iter(self.val))

    def to_string(self):
        return '{%s}' % ','.join(map(str,sweep_set_iter(self.val)))

class SegmentPrinter:
    """Print a poly_ops::detail::segment instance"""
    def __init__(self,val):
        self.val = val

    def to_string(self):
        return '{a=%i, b=%i}' % (int(self.val['a']),int(self.val['b']))

class CachedSegmentPrinter:
    """Print a poly_ops::detail::cached_segment instance"""
    def __init__(self,val):
        self.val = val

    def to_string(self):
        pa = self.val['pa']['_data']
        pb = self.val['pb']['_data']
        return '{a=%i (%i,%i), b=%i (%i,%i)}' % (
            int(self.val['a']),
            int(pa[0]),
            int(pa[1]),
            int(self.val['b']),
            int(pb[0]),
            int(pb[1]))

class CompoundIntPrinter:
    """Print a poly_ops_new::compound_xint instance"""
    def __init__(self,val):
        self.val = val
    
    def to_string(self):
        wcount = int(self.val.type.template_argument(0))
        signed = bool(self.val.type.template_argument(1))
        data = self.val['_data']
        wsize = data.type.target().sizeof
        value = 0
        for i in range(wcount-1,-1,-1):
            value <<= wsize*8
            value |= int(data[i])
        if signed and (int(data[wcount-1]) >> (wsize*8-1)):
            value = -((1 << (wsize*8*wcount)) - value)
        return str(value)

def build_pretty_printer():
    pp = gdb.printing.RegexpCollectionPrettyPrinter('polyops')
    pp.add_printer('point_t','^poly_ops::point_t<.*>$',PointPrinter)
    pp.add_printer('sweep_set','^poly_ops::detail::sweep_set<.*>$',SweepSetPrinter)
    pp.add_printer('segment','^poly(_ops|draw)::detail::segment(<.*>)?$',SegmentPrinter)
    pp.add_printer('cached_segment','^poly(_ops|draw)::detail::cached_segment<.*>$',CachedSegmentPrinter)
    pp.add_printer('compound_xint','^poly_ops_new::compound_xint<.*>$',CompoundIntPrinter)
    return pp
