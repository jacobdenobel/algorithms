import atexit
import matplotlib.pyplot as plt


def plot_positions_interactive(positions, xlim = (-5, 5), ylim = (-5, 5)):
    if not plt.isinteractive():
        plt.ion()
        atexit.register(plt.close)

    axes = plt.gca()
    axes.clear()
    axes.scatter(positions[:, 0], positions[:, 1])
    axes.grid()
    axes.set_xlim(*xlim)
    axes.set_ylim(*ylim)
    plt.pause(0.5)