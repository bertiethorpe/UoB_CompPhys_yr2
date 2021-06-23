#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 04:02:38 2021

@author: bertiethorpe
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

class FreeReturnTrajectory:
    
    def __init__(self, orbital_mass, primary_mass, secondary_mass, secondary_mass_orbit, initial_position, initial_velocity, totaltime, step):
        
        self.m = orbital_mass 
        self.M1 = primary_mass
        self.M2 = secondary_mass
        self.M2_orbit = secondary_mass_orbit
        self.pos_I = initial_position
        self.vel_I = initial_velocity
        self.h = step 
        self.G = 6.67408e-11 
        self.T = totaltime 
    
    def gravityX(self, pos_x, pos_y):
        """function that returns the calculated x-component of the orbiting mass's velocity affected by the central mass,
        using Newtons law of gravitation."""
        
        return ((-self.G * self.M1 * pos_x/((pos_x)**2 + (pos_y)**2)**(3/2)) - (self.G * self.M2 * pos_x/((pos_x**2) + ((pos_y - self.M2_orbit)**2))**(3/2)))
    
    def gravityY(self, pos_x, pos_y):
        """function that returns the calculated y-component of the orbiting mass's velocity affected by the central mass,
        using Newtons law of gravitation."""
        
        return ((-self.G * self.M1 * pos_y/((pos_x**2) + (pos_y**2))**(3/2)) - (self.G * self.M2 * (pos_y - self.M2_orbit)/((pos_x**2) + ((pos_y - self.M2_orbit)**2))**(3/2)))
    
    def energy(self, pos_x, pos_y, vel_x, vel_y):
        """function that returns the total, potential, and kinetic energy of the orbital with time."""
        
        E_k = (1/2) * self.m * ((vel_x)**2 + (vel_y)**2)
        
        E_g = (-self.G * self.M1 * self.m)/(np.sqrt((pos_x)**2 + (pos_y)**2)) + (-self.G * self.M2 * self.m)/(np.sqrt((pos_x)**2 + (pos_y)**2))
        
        E_t = E_k + E_g
        
        return E_k, E_g, E_t
    
    def k_pos_vel(self, pos_x, pos_y, vel_x, vel_y):
        """function that returns multiple arrays of Runge-Kutta 4th Order values for position and velocity of the orbiting mass
        in x and y components. These arrays are later used with the thping functions for velocity and position to determine
        how the orbiting mass moves around the central mass."""
        
        k1_x = vel_x
        k1_y = vel_y
        k1_vx = self.gravityX(pos_x, pos_y)
        k1_vy = self.gravityY(pos_x, pos_y)
        
        k2_x = vel_x + (self.h * k1_vx) / 2
        k2_y = vel_y + (self.h * k1_vy) / 2
        k2_vx = self.gravityX(pos_x + (self.h * k1_x) / 2, pos_y + (self.h * k1_y) / 2)
        k2_vy = self.gravityY(pos_x + (self.h * k1_x) / 2, pos_y + (self.h * k1_y) / 2)
      
        k3_x = vel_x + (self.h * k2_vx) / 2
        k3_y = vel_y + (self.h * k2_vy) / 2
        k3_vx = self.gravityX(pos_x + (self.h * k2_x) / 2, pos_y + (self.h * k2_y) / 2)
        k3_vy = self.gravityY(pos_x + (self.h * k2_x) / 2, pos_y + (self.h * k2_y) / 2) 
       
        k4_x = vel_x + self.h * k3_vx
        k4_y = vel_y + self.h * k3_vy
        k4_vx = self.gravityX(pos_x + self.h * k3_x, pos_y + self.h * k3_y)
        k4_vy = self.gravityY(pos_x + self.h * k3_x, pos_y + self.h * k3_y)
        
        return [k1_x, k2_x, k3_x, k4_x], [k1_y, k2_y, k3_y, k4_y], [k1_vx, k2_vx, k3_vx, k4_vx], [k1_vy, k2_vy, k3_vy, k4_vy]
    
    def thping_position(self, pos_x, pos_y, kpos_xlist, kpos_ylist):
        """function that returns the x and y component of the orbiting mass's position for the next time h."""
        
        pos_x +=  (self.h / 6) * (kpos_xlist[0] + 2 * kpos_xlist[1] + 2 * kpos_xlist[2] + kpos_xlist[3])
        pos_y += (self.h / 6) * (kpos_ylist[0] + 2 * kpos_ylist[1] + 2 * kpos_ylist[2] + kpos_ylist[3])
        
        return pos_x, pos_y
    
    def thping_velocity(self, vel_x, vel_y, kvel_xlist, kvel_ylist):
        """function that returns the x and y component of the orbiting mass's velocity for the next time h."""
        
        vel_x += (self.h / 6) * (kvel_xlist[0] + 2 * kvel_xlist[1] + 2 * kvel_xlist[2] + kvel_xlist[3])
        vel_y += (self.h / 6) * (kvel_ylist[0] + 2 * kvel_ylist[1] + 2 * kvel_ylist[2] + kvel_ylist[3])
       
        return vel_x, vel_y
    
    def time(self, t0 = 0):
        """function that returns an array of times between the initial time and final time with selected time h."""
        
        t = np.linspace(t0, self.T, int(self.T / self.h) +1 )
        
        return t
    
    def simulate_orbit(self):
        """function that returns the x and y components of the orbiting mass's position and velocity throughout the entire simulation
        as arrays used to animate the simulation or used in matplotlib.pyplot to plot a graph of the mass's orbit."""
        
        pos_x, pos_y = self.pos_I[0], self.pos_I[1]
        vel_x, vel_y = self.vel_I[0], self.vel_I[1]
        time_list = self.time()
        rx, ry = np.zeros(len(time_list)), np.zeros(len(time_list))
        rdotx, rdoty = np.zeros(len(time_list)), np.zeros(len(time_list))
        
        for i in np.arange(0, len(time_list)):
            kpos_xlist, kpos_ylist, kvel_xlist, kvel_ylist = self.k_pos_vel(pos_x, pos_y, vel_x, vel_y)
            pos_x, pos_y = self.thping_position(pos_x, pos_y, kpos_xlist, kpos_ylist)
            vel_x, vel_y = self.thping_velocity(vel_x, vel_y, kvel_xlist, kvel_ylist)
            rx[i], ry[i], rdotx[i], rdoty[i] = pos_x, pos_y, vel_x, vel_y
        
        return rx, ry, rdotx, rdoty

day_length = 24 * 60 * 60  
earth_radius = 6.3781e+6
moon_radius = 1.7371e+6
initial_orbit = 5e5 + earth_radius 
earth_mass = 5.972e24 
moon_mass = 7.342e+22
moon_orbit = 3.844e+8
starship_mass = 768 # SpaceX second stage mass
initial_speed = 10.656e3  
multiplier = 20 
        
starship = FreeReturnTrajectory(starship_mass, earth_mass, moon_mass, moon_orbit, [0, -initial_orbit], [initial_speed, 0], (day_length * multiplier), (10))
        
rx, ry, rdotx, rdoty = starship.simulate_orbit()
t = starship.time()
E_k, E_g, E_t = starship.energy(rx, ry, rdotx, rdoty)
        
fig = plt.figure(figsize=(10,7))
gs = fig.add_gridspec(21,2)
        
ax1 = fig.add_subplot(gs[:,1])
p1, = ax1.plot(rx, ry, '--y', linewidth=0.5)
ax1.set_aspect('equal')
ax1.xaxis.set_ticklabels([])
ax1.yaxis.set_ticklabels([])
ax1.set_facecolor('k')
ax1.set_xlim([-2e8,2e8])
ax1.set_ylim([-1e8,5e8])
ax1.add_patch(plt.Circle((0, 0), earth_radius, color='blue'))
ax1.add_patch(plt.Circle((0, moon_orbit), moon_radius, color='0.7'))
        
ax2 = fig.add_subplot(gs[7:19,0])
ax2.set_facecolor('lightgoldenrodyellow')
p2, = ax2.plot(t, E_g, label='Gravitational Energy')
p3, = ax2.plot(t, E_k, label='Kinetic Energy')
p4, = ax2.plot(t, E_t, label='Total Energy')
ax2.set_xlabel('Time (s)', fontsize=9)
ax2.set_ylabel('Energy (J)', fontsize=9)
ax2.legend(ncol=1, fancybox=True, fontsize=6)
        
velocity_slide = plt.axes([0.14,0.81,0.3,0.03], facecolor='k')
orbit_slide = plt.axes([0.14,0.76,0.3,0.03], facecolor='k')
h_slide = plt.axes([0.14,0.71,0.3,0.03], facecolor='k')
        
velocity_factor = Slider(velocity_slide,'Initial Velocity (m/s)',valmin=96,valmax=12e3,valinit=initial_speed,valstep=96,color='y')
velocity_factor.label.set_size(7)
orbit_factor = Slider(orbit_slide,'Initial Orbit (m)',valmin=earth_radius,valmax=8e+6,valinit=initial_orbit,valstep=1000,color='y')
orbit_factor.label.set_size(7)
h_factor = Slider(h_slide,'Time Step (s)',valmin=10,valmax=1000,valinit=100,valstep=10,color='y')
h_factor.label.set_size(7)


    
plt.show()
    




