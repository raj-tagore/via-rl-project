import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import pymunk as pm
from matplotlib.animation import FuncAnimation

class SpringMassSystem:
    def __init__(self, k=10.0, m=1.0, b=0.2, x0=1.0, dt=0.01):
        self.k = k
        self.m = m
        self.b = b
        self.x0 = x0
        self.dt = dt
        
        self.space = pm.Space()
        self.space.gravity = (0.0, -9.81)
        
        self.anchor_body = pm.Body(body_type=pm.Body.STATIC)