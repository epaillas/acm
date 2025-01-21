# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'acm'
copyright = '2025, ACM Topical Group - DESI'
author = 'ACM Topical Group - DESI'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx_design',
    'sphinx.ext.autodoc', # For automatically documenting the functions
    'sphinx.ext.napoleon', # For documenting the parameters of the functions
    'sphinx.ext.intersphinx', # For linking to other packages' documentation
    # 'sphinx.ext.viewcode', # For linking to the source code
    # 'sphinx.ext.linkcode', # For linking to external codes --> Requires a function linkcode_resolve in the conf.py
    'sphinx.ext.autosectionlabel', # For automatically labelling the sections
    'myst_nb', # For including jupyter notebooks
]

myst_enable_extensions = ["dollarmath", "colon_fence", "tasklist"]
myst_enable_checkboxes = True
source_suffix = {
    '.rst': 'restructuredtext',
    '.ipynb': 'myst-nb',
}

# templates_path = ['_templates']
exclude_patterns = []


# -- Autodoc configuration ---------------------------------------------------
# Mock imports that can't be resolved during documentation build
autodoc_mock_imports = [] #'cosmoprimo', 'mockfactory', 'pycorr', 'Corrfunc', 'pyrecon']

napoleon_google_docstring = False
napoleon_numpy_docstring = True

autodoc_preserve_defaults = True # Keep the default values of the parameters instead of replacing them with their values
autoclass_content = 'both' # Include both the class docstring and the __init__ docstring in the documentation
autodoc_member_order = 'bysource' # Order the members by the order in the source code

nb_execution_mode = 'off' # Do not execute the notebooks when building the documentation

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
html_static_path = ['_static']

html_theme = 'sphinx_book_theme'
html_title = 'Alternative Clustering Methods'
# html_logo = "images/logo.svg"
# html_favicon = "images/logo_favicon.svg"
html_show_sourcelink = False # Remove the "view source" link
html_theme_options = {
    # "repository_url": None,
    # "repository_branch": "main",
    # "use_edit_page_button": True,
    # "use_issues_button": True,
    # "use_repository_button": True,
    "use_download_button": True,
    "use_fullscreen_button": False,
    # "logo": {
    #   "image_light": "images/logo.svg",
    #   "image_dark": "images/logo_dark.svg", # Change logo when dark mode is activated
    # },
}
