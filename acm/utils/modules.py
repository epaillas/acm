import importlib


def get_class_from_module(module_path: str, class_name: str) -> type:
    """Dynamically import a class from a module with impotlib."""
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls
