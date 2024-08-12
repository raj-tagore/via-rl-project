import matplotlib.pyplot as plt
import matplotlib.animation as animation
import csv
import os
import time
import numpy as np

global steps_in_view, last_step, last_axis_redraw, cursor_pos, lines
steps_in_view = 1000
last_step = 0
last_axis_redraw = 0
cursor_pos = 0
lines = []

fig, axes = plt.subplots(4,3)
# body - pos, vel, acc; ee - pos, vel, acc, contact, rest_len, stiffness; surface - pos, vel, acc
step = []
state = [[],[],[],[],[],[],[],[],[],[],[],[]]

file = open('current_run.csv', 'r')

def read_line(file, cursor_pos):
    file.seek(cursor_pos)
    row = file.readline().strip().split(',')
    cursor_pos = file.tell()
    return row, cursor_pos

def init_plot():
    global lines
    lines = []
    for i, ax in enumerate(np.nditer(axes, flags=['refs_ok'])):
        ax = ax.item()
        line, = ax.plot(step, state[i])
        lines.append(line)
    return lines

def update_plot(frame):
    global last_modified, steps_in_view, last_step, last_axis_redraw, cursor_pos, lines
    
    # read data from file
    row, cursor_pos = read_line(file, cursor_pos)
    
    # if all data is read, keep graph static
    if row[0] == '':
        return lines
    
    # clear data if new episode
    if int(row[0]) < last_step:
        for i in range(12):
            state[i].clear()
        step.clear()
    # pause if sim is paused
    elif int(row[0]) < last_step+0:
        return lines
                
    last_step = int(row[0])
    
    # append new data to lists
    step.append(int(row[0]))
    for i in range(1, 13):
        state[i-1].append(float(row[i]))
        
    # if size of lists exceeds the view, remove the oldest data
    if len(step) > steps_in_view:
        step.pop(0)
        for i in range(12):
            state[i].pop(0)
    
    # update lines        
    for i, line in enumerate(lines):
        line.set_xdata(step)
        line.set_ydata(state[i])
    
    # reset x-axis limits
    if step[-1] > last_axis_redraw+100:    
        for i, ax in enumerate(np.nditer(axes, flags=['refs_ok'])):
            ax = ax.item()
            ax.set_xlim(min(step)-50, max(step)+50)
            ax.set_ylim(min(state[i])-10, max(state[i])+10)
            ax.figure.canvas.draw() 
        last_axis_redraw = step[-1]
        
    return lines


ani = animation.FuncAnimation(fig, 
                              update_plot, 
                              init_func=init_plot, 
                              interval=1, 
                              blit=True,
                              cache_frame_data=False)
plt.tight_layout()
plt.show()