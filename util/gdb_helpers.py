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

class Plotter(gdb.Command):
    """Display the lines represented by a vector of poly_ops::detail::loop_point
    instances."""

    def __init__(self):
        super().__init__('plot_lpoints',gdb.COMMAND_DATA,gdb.COMPLETE_SYMBOL)
        self.proc = None
    
    def invoke(self,argument,from_tty):
        if not argument: raise gdb.GdbError('an argument is required')

        data = gdb.parse_and_eval('({}).data()'.format(argument))
        size = int(gdb.parse_and_eval('({}).size()'.format(argument)))

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

def build_pretty_printer():
    pp = gdb.printing.RegexpCollectionPrettyPrinter('polyops')
    pp.add_printer('point_t','^poly_ops::point_t<.*>$',PointPrinter)
    return pp