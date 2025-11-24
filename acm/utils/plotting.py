from functools import wraps
import matplotlib.pyplot as plt

def set_plot_style(func):
    """Decorator to set the plotting style to acm standard."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        return func(*args, **kwargs)
    return wrapper