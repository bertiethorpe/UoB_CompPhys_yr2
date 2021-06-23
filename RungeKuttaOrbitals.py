#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 22:09:27 2021

@author: bertiethorpe
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

np.seterr(divide='ignore', invalid='ignore')

G = 6.67430e-11
M_earth = 5.9742e+24
M_moon = 7.36e+22
moon_average_orbit = 3.844e+8
m_starship = 1.4e+6
r_earth = 6.3781e+6
r_moon = 1.7374e+6

h=100
T=2000000
N = 2000

pos_x = np.zeros(N)
pos_y = np.zeros(N)
vel_x = np.zeros(N)
vel_y = np.zeros(N)


pos_I = [r_earth + 6e+7, 0]
vel_I = [0, 10000]

x0 = r_earth + 6e+7
y0 = 0
vx0 = 0
vy0 = 1000

def GravityX(pos_x, pos_y):

    return ((-G * M_earth * pos_x/((pos_x**2) + (pos_y**2))**(3/2)) + (-G * M_moon * pos_x/((pos_x**2) + ((pos_y - moon_average_orbit)**2))**(3/2)))

def GravityY(pos_x, pos_y):
    
    return ((-G * M_earth * pos_y/((pos_x**2) + (pos_y**2))**(3/2)) + (-G * M_moon * (pos_y - moon_average_orbit)/((pos_x**2) + ((pos_y - moon_average_orbit)**2))**(3/2)))
    
def RK4(pos_x, pos_y, vel_x, vel_y):
    
    for i in np.arange(N):
        
        k1_x = vel_x
        k1_y = vel_y
        k1_vx = GravityX(pos_x, pos_y)
        k1_vy = GravityY(pos_x, pos_y)
        
        k2_x = vel_x + (h * k1_vx)/2
        k2_y = vel_y + (h * k1_vy)/2
        k2_vx = GravityX(pos_x + (h * k1_x)/2, pos_y + (h * k1_y)/2)
        k2_vy = GravityY(pos_x + (h * k1_x)/2, pos_y + (h * k1_y)/2)
    
        k3_x = vel_x + (h * k2_vx)/2
        k3_y = vel_y + (h * k2_vy)/2
        k3_vx = GravityX(pos_x + (h * k2_x)/2, pos_y + (h * k2_y)/2)
        k3_vy = GravityY(pos_x + (h * k2_x)/2, pos_y + (h * k2_y)/2)
        
        k4_x = vel_x + (h * k3_vx)
        k4_y = vel_y + (h * k3_vy)
        k4_vx = GravityX(pos_x + (h * k3_x), pos_y + (h * k3_y))
        k4_vy = GravityY(pos_x + (h * k3_x), pos_y + (h * k3_y))
        
        pos_x = pos_x + h/6 * (k1_x + 2*k2_x + 2*k3_x + k4_x)
        pos_y = pos_y + h/6 * (k1_y + 2*k2_y + 2*k3_y + k4_y)
        vel_x = vel_x + h/6 * (k1_vx + 2*k2_vx + 2*k3_vx + k4_vx)
        vel_y = vel_y + h/6 * (k1_vy + 2*k2_vy + 2*k3_vy + k4_vy)
        
    return pos_x, pos_y, vel_x, vel_y


pos_x, pos_y, vel_x, vel_y = RK4(pos_x, pos_y, vel_x, vel_y) 
  
fig, ax = plt.subplots()
ax.plot(pos_x, pos_y)
plt.show()


        
        


-(self.G * self.M1 * pos_x) / ((pos_x ** 2 + pos_y ** 2) ** (3 / 2))


-(self.G * self.M1 * pos_y) / ((pos_x ** 2 + pos_y ** 2) ** (3 / 2))




        
