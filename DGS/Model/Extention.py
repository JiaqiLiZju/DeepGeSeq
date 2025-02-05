"""
Extention utilities for loading and instantiating extended modules and models
"""

import os, sys, importlib, types

def module_from_file(path):
    """
    Load a module created based on a Python file path.

    Parameters
    ----------
    path : str
        Path to the model architecture file.

    Returns
    -------
    The loaded module

    """
    parent_path, module_file = os.path.split(path)
    loader = importlib.machinery.SourceFileLoader(
        module_file[:-3], path)
    module = types.ModuleType(loader.name)
    loader.exec_module(module)
    return module


def module_from_dir(path):
    """
    This method expects that you pass in the path to a valid Python module,
    where the `__init__.py` file already imports the model class.
    (e.g. `__init__.py` contains the line `from <model_class_file> import
    <ModelClass>`).

    Parameters
    ----------
    path : str
        Path to the Python module containing the model class.

    Returns
    -------
    The loaded module
    """
    parent_path, module_dir = os.path.split(path)
    sys.path.insert(0, parent_path)
    return importlib.import_module(module_dir)


def load_module(import_model_from, model_class_name=None):

    module = None
    if os.path.isdir(import_model_from):
        module = module_from_dir(import_model_from)
    else:
        module = module_from_file(import_model_from)

    if model_class_name is None:
        return module
    
    model_class = getattr(module, model_class_name)
    return module, model_class
