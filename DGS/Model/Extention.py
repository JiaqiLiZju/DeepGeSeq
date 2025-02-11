"""Extension utilities for dynamic model loading in DGS.

This module provides utilities for dynamically loading and instantiating model modules
and classes from Python files or directories. It supports both single-file models and
complete Python packages.

Functions
---------
module_from_file(path)
    Load a module from a single Python file.

module_from_dir(path)
    Load a module from a directory containing an __init__.py file.

load_module(import_model_from, model_class_name)
    High-level function to load either a file or directory module.

Notes
-----
The module loading functions are designed to be flexible and support different
module organization patterns. They can handle:
- Single file models
- Package models with __init__.py
- Models with custom import structures
"""

import os, sys, importlib, types

def module_from_file(path):
    """Load a Python module from a file path.

    This function creates a module object from a Python file without requiring
    the file to be in the Python path. It uses Python's import machinery directly.

    Parameters
    ----------
    path : str
        Absolute or relative path to the Python file to load.

    Returns
    -------
    types.ModuleType
        The loaded module object.

    Examples
    --------
    >>> module = module_from_file('path/to/model.py')
    >>> model_class = getattr(module, 'ModelClass')
    >>> model = model_class()

    Notes
    -----
    - The file must be a valid Python module
    - The function uses SourceFileLoader from importlib.machinery
    - The module name will be the filename without .py extension
    """
    parent_path, module_file = os.path.split(path)
    loader = importlib.machinery.SourceFileLoader(
        module_file[:-3], path)
    module = types.ModuleType(loader.name)
    loader.exec_module(module)
    return module


def module_from_dir(path):
    """Load a Python module from a directory.

    This function loads a module from a directory containing an __init__.py file.
    The directory must be a valid Python package.

    Parameters
    ----------
    path : str
        Path to the directory containing the Python module.

    Returns
    -------
    module
        The loaded module object.

    Examples
    --------
    >>> module = module_from_dir('path/to/model_package')
    >>> model_class = getattr(module, 'ModelClass')
    >>> model = model_class()

    Notes
    -----
    - The directory must contain an __init__.py file
    - The __init__.py should import all relevant classes
    - The function temporarily modifies sys.path
    """
    parent_path, module_dir = os.path.split(path)
    sys.path.insert(0, parent_path)
    return importlib.import_module(module_dir)


def load_module(import_model_from, model_class_name=None):
    """Load a module and optionally a specific class from it.

    This is a high-level function that can load modules from either files or
    directories. It provides a unified interface for module loading.

    Parameters
    ----------
    import_model_from : str
        Path to either a Python file or a directory containing a Python package.
    model_class_name : str, optional
        Name of the specific class to load from the module.
        If None, returns only the module.

    Returns
    -------
    module : types.ModuleType
        The loaded module object.
    model_class : type, optional
        The model class if model_class_name was specified.

    Examples
    --------
    >>> # Load just the module
    >>> module = load_module('path/to/model.py')
    >>> 
    >>> # Load module and specific class
    >>> module, model_class = load_module('path/to/model.py', 'ModelClass')
    >>> model = model_class()

    Notes
    -----
    - Can handle both file and directory imports
    - Returns either module or (module, class) tuple
    - Raises AttributeError if model_class_name is not found in module
    """
    module = None
    if os.path.isdir(import_model_from):
        module = module_from_dir(import_model_from)
    else:
        module = module_from_file(import_model_from)

    if model_class_name is None:
        return module
    
    model_class = getattr(module, model_class_name)
    return module, model_class
