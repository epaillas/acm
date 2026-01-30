import importlib

def get_class_from_module(module_path, class_name):
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls