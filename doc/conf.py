
#import os
#import sys
#sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'polyops'
copyright = '2023, Rouslan Korneychuk'
author = 'Rouslan Korneychuk'

release = '0.1'


# -- General configuration ---------------------------------------------------

extensions = ['breathe']

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

pygments_style = 'sas'
html_theme = 'nature'
html_theme_options = {
    'sidebarwidth': 300}
html_sidebars = {'**':['searchbox.html','globaltoc.html']}

toc_object_entries = True
toc_object_entries_show_parents='hide'

cpp_index_common_prefix = ['poly_ops::']
cpp_maximum_signature_line_length = 120

breathe_default_project = 'polyops'
breathe_show_include = False
breathe_use_cpp_namespace = True

