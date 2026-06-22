import importlib


def get_class_from_module(module_path: str, class_name: str) -> type:
    """Dynamically import a class from a module with impotlib."""
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls


def check_installed(name: str) -> bool:
    """Check if a package is installed on-fly."""
    try:
        importlib.import_module(name)
    except ImportError:
        return False
    else:
        return True
