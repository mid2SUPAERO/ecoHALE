# -*- coding: utf-8 -*-
# This file is execfile()d with the current directory set to its
# containing dir.
import sys
import os
import importlib
from mock import Mock
from openmdao.docs.config_params import MOCK_MODULES
# from openaerostruct.docs._utils.patch import do_monkeypatch
# from openaerostruct.docs._utils.generate_sourcedocs import generate_docs

import openaerostruct
openaerostruct_path = os.path.split(os.path.abspath(openaerostruct.__file__))[0]
sys.path.insert(0, os.path.join(openaerostruct_path, 'docs', '_exts'))

sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('.'))


# Only mock the ones that don't import.
for mod_name in MOCK_MODULES:
    try:
        importlib.import_module(mod_name)
    except ImportError:
        sys.modules[mod_name] = Mock()

# start off running the numpydoc monkeypatch to keep options/parameters/returns
# usable in docstring for autodoc.
from numpydoc.docscrape_sphinx import SphinxDocString
from numpydoc.docscrape import NumpyDocString, Reader
import textwrap

# start off running the monkeypatch to keep options/parameters
# usable in docstring for autodoc.


def __init__(self, docstring, config={}):
    """
    init
    """
    docstring = textwrap.dedent(docstring).split('\n')

    self._doc = Reader(docstring)
    self._parsed_data = {
        'Signature': '',
        'Summary': [''],
        'Extended Summary': [],
        'Parameters': [],
        'Options': [],
        'Returns': [],
        'Yields': [],
        'Raises': [],
        'Warns': [],
        'Other Parameters': [],
        'Attributes': [],
        'Methods': [],
        'See Also': [],
        'Notes': [],
        'Warnings': [],
        'References': '',
        'Examples': '',
        'index': {}
    }

    try:
        self._parse()
    except ParseError as e:
        e.docstring = orig_docstring
        raise

    # In creation of docs, remove private Attributes (beginning with '_')
    # with a crazy list comprehension
    self._parsed_data["Attributes"][:] = [att for att in self._parsed_data["Attributes"]
                                          if not att[0].startswith('_')]


def _parse(self):
    """
    parse
    """
    self._doc.reset()
    self._parse_summary()

    sections = list(self._read_sections())
    section_names = set([section for section, content in sections])

    has_returns = 'Returns' in section_names
    has_yields = 'Yields' in section_names
    # We could do more tests, but we are not. Arbitrarily.
    if has_returns and has_yields:
        msg = 'Docstring contains both a Returns and Yields section.'
        raise ValueError(msg)

    for (section, content) in sections:
        if not section.startswith('..'):
            section = (s.capitalize() for s in section.split(' '))
            section = ' '.join(section)
            if self.get(section):
                msg = ("The section %s appears twice in the docstring." %
                       section)
                raise ValueError(msg)

        if section in ('Parameters', 'Options', 'Params', 'Returns', 'Yields', 'Raises',
                       'Warns', 'Other Parameters', 'Attributes',
                       'Methods'):
            self[section] = self._parse_param_list(content)
        elif section.startswith('.. index::'):
            self['index'] = self._parse_index(section, content)
        elif section == 'See Also':
            self['See Also'] = self._parse_see_also(content)
        else:
            self[section] = content


def __str__(self, indent=0, func_role="obj"):
    """
    our own __str__
    """
    out = []
    out += self._str_signature()
    out += self._str_index() + ['']
    out += self._str_summary()
    out += self._str_extended_summary()
    out += self._str_param_list('Parameters')
    out += self._str_options('Options')
    out += self._str_returns()
    for param_list in ('Other Parameters', 'Raises', 'Warns'):
        out += self._str_param_list(param_list)
    out += self._str_warnings()
    out += self._str_see_also(func_role)
    out += self._str_section('Notes')
    out += self._str_references()
    out += self._str_examples()
    for param_list in ('Attributes', 'Methods'):
        out += self._str_member_list(param_list)
    out = self._str_indent(out, indent)
    return '\n'.join(out)


def _str_options(self, name):
    """
    """
    out = []
    if self[name]:
        out += self._str_field_list(name)
        out += ['']
        for param, param_type, desc in self[name]:
            if param_type:
                out += self._str_indent(['**%s** : %s' % (param.strip(),
                                                          param_type)])
            else:
                out += self._str_indent(['**%s**' % param.strip()])
            if desc:
                out += ['']
                out += self._str_indent(desc, 8)
            out += ['']
    return out


# Do the actual patch switchover to these local versions
def do_monkeypatch():
    NumpyDocString.__init__ = __init__
    SphinxDocString._str_options = _str_options
    SphinxDocString._parse = _parse
    SphinxDocString.__str__ = __str__

do_monkeypatch()


# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.doctest',
              'sphinx.ext.coverage',
              'sphinx.ext.mathjax',
              'sphinx.ext.viewcode',
              'sphinx.ext.githubpages',
              'numpydoc',
              'embed_code',
              'embed_options']

# directories for which to generate sourcedocs
packages = [
    'aerodynamics',
    'functionals',
    'geometry',
    'integration',
    'structures',
    'transfer',
]

#begin source doc stuff
IGNORE_LIST = [
        'docs', 'tests', 'fortran', 'utils'
    ]
# this function is used to create the entire directory structure
# of our source docs, as well as writing out each individual rst file.


def generate_docs(dir, top, packages):
    """
    generate_docs
    """
    index_top = """:orphan:

.. _source_documentation:

********************
Source Docs
********************

.. toctree::
   :titlesonly:
   :maxdepth: 1

"""
    package_top = """
.. toctree::
    :maxdepth: 1

"""

    ref_sheet_bottom = """
   :members:
   :undoc-members:
   :special-members: __init__, __contains__, __iter__, __setitem__, __getitem__
   :show-inheritance:
   :inherited-members:

.. toctree::
   :maxdepth: 1
"""

    docs_dir = os.path.dirname(dir)

    doc_dir = os.path.join(docs_dir, "_srcdocs")
    if os.path.isdir(doc_dir):
        import shutil
        shutil.rmtree(doc_dir)

    if not os.path.isdir(doc_dir):
        os.mkdir(doc_dir)

    packages_dir = os.path.join(doc_dir, "packages")
    if not os.path.isdir(packages_dir):
        os.mkdir(packages_dir)

    # look for directories in the openaero level, one up from docs
    # those directories will be the openmdao packages
    # auto-generate the top-level index.rst file for _srcdocs, based on
    # openaerostruct packages:

    # to improve the order that the user sees in the source docs, put
    # the important packages in this list explicitly. Any new ones that
    # get added will show up at the end.

    # everything in openaerostruct dir that isn't discarded is appended as a source package.
    for listing in os.listdir(os.path.join(top)):
        if os.path.isdir(os.path.join("..", listing)):
            if listing not in IGNORE_LIST and listing not in packages:
                packages.append(listing)

    # begin writing the '_srcdocs/index.rst' file at mid  level.
    index_filename = os.path.join(doc_dir, "index.rst")
    index = open(index_filename, "w")
    index.write(index_top)

    # auto-generate package header files (e.g. 'openaerostruct.core.rst')
    for package in packages:
        # a package is e.g. openaerostruct.core, that contains source files
        # a sub_package, is a src file, e.g. openaerostruct.core.component
        sub_packages = []
        package_filename = os.path.join(packages_dir,
                                        "openaerostruct." + package + ".rst")
        package_name = "openaerostruct." + package

        # the sub_listing is going into each package dir and listing what's in it
        for sub_listing in sorted(os.listdir(os.path.join(dir, package.replace('.','/')))):
            # don't want to catalog files twice, nor use init files nor test dir
            if (os.path.isdir(sub_listing) and sub_listing != "tests") or \
               (sub_listing.endswith(".py") and not sub_listing.startswith('_')):
                # just want the name of e.g. dataxfer not dataxfer.py
                sub_packages.append(sub_listing.rsplit('.')[0])

        if len(sub_packages) > 0:
            # continue to write in the top-level index file.
            # only document non-empty packages -- to avoid errors
            # (e.g. at time of writing, doegenerators, drivers, are empty dirs)

            # specifically don't use os.path.join here.  Even windows wants the
            # stuff in the file to have fwd slashes.
            index.write("   packages/openaerostruct." + package + "\n")

            # make subpkg directory (e.g. _srcdocs/packages/core) for ref sheets
            package_dir = os.path.join(packages_dir, package)
            os.mkdir(package_dir)

            # create/write a package index file: (e.g. "_srcdocs/packages/openaerostruct.core.rst")
            package_file = open(package_filename, "w")
            package_file.write(package_name + "\n")
            package_file.write("-" * len(package_name) + "\n")
            package_file.write(package_top)

            for sub_package in sub_packages:
                SKIP_SUBPACKAGES = ['__pycache__']
                # this line writes subpackage name e.g. "core/component.py"
                # into the corresponding package index file (e.g. "openaerostruct.core.rst")
                if sub_package not in SKIP_SUBPACKAGES:
                    # specifically don't use os.path.join here.  Even windows wants the
                    # stuff in the file to have fwd slashes.
                    package_file.write("    " + package + "/" + sub_package + "\n")

                    # creates and writes out one reference sheet (e.g. core/component.rst)
                    ref_sheet_filename = os.path.join(package_dir, sub_package + ".rst")
                    ref_sheet = open(ref_sheet_filename, "w")

                    # get the meat of the ref sheet code done
                    filename = sub_package + ".py"
                    ref_sheet.write(".. index:: " + filename + "\n\n")
                    ref_sheet.write(".. _" + package_name + "." +
                                    filename + ":\n\n")
                    ref_sheet.write(filename + "\n")
                    ref_sheet.write("-" * len(filename) + "\n\n")
                    ref_sheet.write(".. automodule:: " + package_name + "." + sub_package)

                    # finish and close each reference sheet.
                    ref_sheet.write(ref_sheet_bottom)
                    ref_sheet.close()

            # finish and close each package file
            package_file.close()

    # finish and close top-level index file
    index.close()


generate_docs("..", "../..", packages)

# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
needs_sphinx = '1.6'

numpydoc_show_class_members = False

# The master toctree document.
#master_doc = 'index'

# General information about the project.
project = u'OpenAeroStruct'
copyright = u'2017, John Jasa, Dr. John Hwang'
author = u'John Jasa, Dr. John Hwang'

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = '0.3.2'
# The full version, including alpha/beta/rc tags.
release = '0.3.2'

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'


# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = '_theme'

# Add any paths that contain custom themes here, relative to this directory.
html_theme_path = ['.']
html_static_path = ['_static']
html_context = {
    'css_files': ['_static/style.css', ],
}

# # The name of an image file (relative to this directory) to place at the top
# # of the sidebar.
# html_logo = '_static/OpenMDAO_Logo.png'
#
# # The name of an image file (within the static path) to use as favicon of the
# # docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# # pixels large.
# html_favicon = '_static/OpenMDAO_Favicon.ico'

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
html_last_updated_fmt = '%b %d, %Y'

# Output file base name for HTML help builder.
htmlhelp_basename = 'OpenAeroStructdoc'

#Customize sidebar
html_sidebars = {
   '**': ['globaltoc.html', 'searchbox.html']
}

# The master toctree document.
master_doc = 'index'
# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'openaerostruct', u'OpenAeroStruct Documentation',
     [author], 1)
]