import matplotlib.pyplot as plt
import matplotlib.animation as animation
import csv
import os
import time
import numpy as np

global steps_in_view, last_step, cursor_pos, lines
steps_in_view = 1000
last_step = 0
cursor_pos = 0
last_cursor_pos = 0
lines = []

fig, axes = plt.subplots(4, 3)
# body - pos, vel, acc; ee - pos, vel, acc, contact, rest_len, stiffness; surface - pos, vel, acc
step = []
state = [[],[],[],[],[],[],[],[],[],[],[],[]]

file = open('current_run.csv', 'r')

def read_line(file, cursor_pos):
    file.seek(cursor_pos)
    row = file.readline().strip().split(',')
    cursor_pos = file.tell()
    return row, cursor_pos

def update_plot(frame):
    global last_modified, steps_in_view, last_step, cursor_pos
    row, cursor_pos = read_line(file, cursor_pos)
    if int(row[0]) < last_step:
        for i in range(12):
            state[i].clear()
        step.clear()
    elif int(row[0]) < last_step+10:
        return
    last_step = int(row[0])
    step.append(int(row[0]))
    for i in range(1, 13):
        state[i-1].append(float(row[i]))
    if len(step) > steps_in_view:
        step.pop(0)
        for i in range(12):
            state[i].pop(0)
    for i, ax in enumerate(np.nditer(axes, flags=['refs_ok'])):
        ax = ax.item()
        ax.clear()
        ax.plot(step, state[i])
            
ani = animation.FuncAnimation(fig, update_plot, interval=1)
plt.tight_layout()
plt.show()
        
            
        
