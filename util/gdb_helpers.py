import subprocess
import json
import os.path
import sys
import gdb
import gdb.printing

MAX_PLOT_SIZE = 400

PLOTTER = os.path.join(os.path.dirname(__file__),'plotter.py')

def cpp_loop_val_to_py(x,line_state_t_names):
    data = x['data']['_data']
    return {
        'p': (int(data[0]),int(data[1])),
        'next': int(x['next']),
        'state': line_state_t_names.get(x['aux']['desc']['state'],'<invalid>'),
        'loop_i': int(x['aux']['loop_index'])}

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

        line_state_t_names = {}
        for name in (
                'undef','check','discard','discard_rev','keep','keep_rev',
                'anchor_undef','anchor_discard','anchor_discard_rev',
                'anchor_keep','anchor_keep_rev'):
            val = gdb.lookup_global_symbol('poly_ops::detail::line_state::' + name)
            line_state_t_names[val] = name

        points = [cpp_loop_val_to_py(data[i],line_state_t_names) for i in range(size)]

        if self.proc is None or self.proc.poll() is not None:
            self.proc = subprocess.Popen([sys.executable,PLOTTER],stdin=subprocess.PIPE,text=True)

        assert self.proc.stdin is not None
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
        return sweep_set_iter(self.val)
    
    def children(self):
        for i,val in enumerate(self):
            yield str(i),val

    def display_hint(self):
        return 'array'

    def to_string(self):
        return 'sweep_set instance'

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
    """Print a poly_ops::large_ints::compound_xint instance"""
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

class DrawEventPrinter:
    """Print a poly_ops::draw::detail::event instance"""
    def __init__(self,val):
        self.val = val
    
    def to_string(self):
        try:
            type = self.val['type']
            event_ns = 'poly_ops::draw::detail::event_type_t::'
            ab = self.val['ab']
            a = int(ab['a'])
            b = int(ab['b'])
            swap_sym = gdb.lookup_global_symbol(event_ns + 'swap')
            if swap_sym is None:
                return 'error: cannot find "{event_ns}swap"'
            if type == swap_sym.value():
                y_intr = self.val['intr_y']
                return f'swap sweep nodes {a} and {b} at {y_intr}'
            else:
                forward_sym = gdb.lookup_global_symbol(event_ns + 'forward')
                if forward_sym is None:
                    return 'error: cannot find "{event_ns}forward"'
                t_str = 'forward' if type == forward_sym.value() else 'backward'
                sn = int(self.val['sweep_node'])
                return f'{t_str} line {a} - {b}, sweep_node {sn}'
        except Exception as e:
            return str(e)

class MiniFlatSetPrinter:
    """Print a poly_ops::detail::mini_flat_set instance"""
    def __init__(self,val):
        self.val = val
    
    def to_string(self):
        try:
            size = self.val['_size']
            items = []
            if size == 1:
                items.append(str(self.val['u']['item']))
            elif size > 0:
                data = self.val['u']['data'].dereference()['itmes']
                for i in range(size):
                    items.append(str(data[i]))
            
            return '{%s}' % ','.join(items)
        except Exception as e:
            return str(e)

class PointAndOriginPrinter:
    """Print a poly_ops::point_and_origin instance"""
    def __init__(self,val):
        self.val = val

    def to_string(self):
        orig_dbg = self.val['original_points']
        ptr = orig_dbg['_M_ptr']
        orig = [str(ptr[i]) for i in range(int(orig_dbg['_M_extent']['_M_extent_value']))]
        return '{%s, {%s}}' % (str(self.val['p']),','.join(orig))

def build_pretty_printer():
    pp = gdb.printing.RegexpCollectionPrettyPrinter('polyops')
    pp.add_printer('point_t','^poly_ops::point_t<.*>$',PointPrinter)
    pp.add_printer('sweep_set','^poly_ops::detail::sweep_set<.*>$',SweepSetPrinter)
    pp.add_printer('segment','^poly_ops(::draw)?::detail::segment(<.*>)?$',SegmentPrinter)
    pp.add_printer('cached_segment','^poly_ops(::draw)?::detail::cached_segment<.*>$',CachedSegmentPrinter)
    pp.add_printer('compound_xint','^poly_ops::large_ints::compound_xint<.*>$',CompoundIntPrinter)
    pp.add_printer('draw_event','^poly_ops::draw::detail::event<.*>$',DrawEventPrinter)
    pp.add_printer('mini_flat_set','^poly_ops::detail::mini_flat_set<.*>$',MiniFlatSetPrinter)
    pp.add_printer('point_t','^poly_ops::point_and_origin<.*>$',PointAndOriginPrinter)
    return pp
