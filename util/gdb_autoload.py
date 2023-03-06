import sys
import os.path
import gdb.printing

sys.path.insert(0,os.path.dirname(__file__))

import gdb_helpers

gdb_helpers.Plotter()

gdb.printing.register_pretty_printer(
    gdb.current_objfile(),
    gdb_helpers.build_pretty_printer())
