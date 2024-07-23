import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

# Create a figure and axes for the plots
fig, (ax1, ax2) = plt.subplots(2, 1)

# Initialize lists to store data
x_data = []
y_data = []
scatter_data = []

# Function to update data
def update_data():
    global x_data, y_data, scatter_data
    if len(x_data) > 50:
        x_data.pop(0)
        y_data.pop(0)
        scatter_data.pop(0)
    x_data.append(len(x_data))
    y_data.append(random.randint(0, 100))
    scatter_data.append((len(x_data), random.randint(0, 100)))

# Function to animate the line plot
def animate_line(i):
    update_data()
    ax1.clear()
    ax1.plot(x_data, y_data)
    ax1.set_title('Line Plot')
    ax1.set_xlim([0, max(50, len(x_data))])
    ax1.set_ylim([0, 100])

# Function to animate the scatter plot
def animate_scatter(i):
    update_data()
    ax2.clear()
    x, y = zip(*scatter_data)
    ax2.scatter(x, y)
    ax2.set_title('Scatter Plot')
    ax2.set_xlim([0, max(50, len(x))])
    ax2.set_ylim([0, 100])

# Set up the animation
ani1 = animation.FuncAnimation(fig, animate_line, interval=500)
ani2 = animation.FuncAnimation(fig, animate_scatter, interval=500)

# Display the plot
plt.tight_layout()
plt.show()
