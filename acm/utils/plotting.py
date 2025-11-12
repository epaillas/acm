from functools import wraps
import matplotlib.pyplot as plt

# decorator to set plotting style
def set_plot_style(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        return func(*args, **kwargs)
    return wrapper