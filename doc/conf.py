
#import os
#import sys
#sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'polyops'
copyright = '2023, Rouslan Korneychuk'
author = 'Rouslan Korneychuk'

release = '0.1'


# -- General configuration ---------------------------------------------------

extensions = []

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

pygments_style = 'sas'
html_theme = 'nature'
html_sidebars = {'**':['searchbox.html','globaltoc.html']}

toc_object_entries = True
toc_object_entries_show_parents='hide'

cpp_index_common_prefix = ['poly_ops::']
