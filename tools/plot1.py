import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random

# Create a figure and an axis
fig, ax = plt.subplots()
xdata, ydata = [], []
ln, = plt.plot([], [], 'b-')  # 'b-' means blue line

# Initialization function to set up the plot's background
def init():
    ax.set_xlim(2, 10)
    ax.set_ylim(0, 10)
    return ln,

# Update function for each frame
def update(frame):
    print(frame)
    xdata.append(frame)
    ydata.append(random.uniform(0, 10))  # Simulate real-time data
    if len(xdata) > 10:
        xdata.pop(0)
        ydata.pop(0)
    ln.set_data(xdata, ydata)
    ax.set_xlim(max(0,frame-10), frame)
    return ln,

# Create the animation object
ani = FuncAnimation(fig, update, frames=range(100),
                    init_func=init, blit=False)

# Show the plot
plt.show()
