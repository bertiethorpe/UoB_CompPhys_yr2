#!/usr/bin/env python
# coding: utf-8

# # Second Year Computational Physics
# ## Exercise 2

# The deadline for this exercise is Monday 7th December 2020 at 12:30 p.m.  Your Jupyter notebook file (.ipynb) should be uploaded into Blackboard at the appropriate point in the Second Year Lab DLM (PHY2DLM_2020: Physics DLM Year 2 2020) course. *S. Hanna*

# ### Objectives of the exercise
# 
# * To become familiar with some basic tools for solving ordinary differential equations;
# * To apply the Euler method to a 1D free fall problem with varying air resistance;

# In this second exercise, as for the first, you will be required to submit a single notebook that addresses all of the points below.  You will need to complete the code cells with working Python code, and fill in the answers in the text cells. No report will be required.

# ## Problem: Free-fall with fixed or varying drag
# On 14th October 2012, Felix Baumgartner set the world record for falling from a great height.  He jumped from a helium balloon at a height of 39045 m, fell for 4 minutes and 19 seconds and reached a maximum speed of 373 m s$^{-1}$. In this problem, you will solve the equations of motion for a free-falling object, and use your program to confirm, or otherwise, Felix Baumgartner's statistics. [In fact Felix BaumGartner's record stood until 24th October 2014, when Alan Eustace (a senior Google vice president) jumped from 41419 m, reaching a maximum speed of 367m s$^{-1}$. A sonic boom was heard by observers on the ground.  However, Eustace's attempt was very low-key compared with Baumgartner, and it is the latter who tends to be remembered.]
# 
# For a projectile travelling at speed through the air, the air resistance is proportional to the square of the velocity and acts in the opposite direction.  i.e. 
# 
# $$\mathbf{f} = -kv^2\hat{\mathbf{v}} = -k|\mathbf{v}|{\mathbf{v}}.$$
# 
# The  constant, $k$, is given by:
# $$
# k = \frac{C_\mathrm{d}\rho_0 A}{2}\qquad(1)
# $$
# 
# in which $C_\mathrm{d}$ is the drag coefficient ($\sim0.47$ for a sphere; $\sim1.0-1.3$ for a sky diver or ski jumper), $A$ is the cross sectional area of the projectile and $\rho_0$ is the air density ($\sim1.2$ kg m$^{-3}$ at ambient temperature and pressure).  The resultant acceleration depends only on the weight and the air resistance:
# 
# $$m\mathbf{a} = m\mathbf{g}+\mathbf{f}$$
# 
# In this problem, the acceleration is varying, so application of Newton's second law produces a second order ODE to solve.  As illustrated in lectures, we can do this if we separate it into two first order equations, one for the derivative of the velocity, the other for the position:
# 
# $$\displaystyle m\frac{dv_y}{dt} = -mg-k\big|v_y\big|v_y\quad;\quad\frac{dy}{dt} = v_y\quad(2)$$
# 
# The $y$ coordinate is taken vertically upwards.   Euler's method for solving:
# 
# $$\frac{dy}{dt} = f(y,t)$$ is summarised by:
# 
# $$
# y_{n+1} = y_n + \Delta t.f\left(y_n,t_n\right)\qquad;\qquad t_{n+1} = t_n + \Delta t \label{eq:euler}
# $$
# 
# in which we are determining $y$ and $t$ at the $(n+1)$th step from their values at the $n$th step.  Applying this scheme to Eqs. (1) and (2), we obtain:
# 
# \begin{eqnarray*}
# \label{eq:eu1}v_{y,n+1} &=&v_{y,n} - \Delta t\left(g+\frac{k}{m}\big|v_{y,n}\big|v_{y,n}\right)\quad(3)\\[2ex]
# y_{n+1} &=& y_n + \Delta t.v_{y,n}\quad(4)\\[2ex]
# \label{eq:eu3}t_{n+1} &=& t_n + \Delta t\quad(5)\\[1ex]\nonumber
# \end{eqnarray*}
# 
# If we provide the initial conditions i.e. $y_0$ and $v_{y,0}$, we can use the above scheme repeatedly to find $y$ and $v_y$ for all $t$.
# 
# Attempt the following programming tasks and make sure you answer all the points raised in the appropriate cells in your notebook.
# 
# ---

# #### Part (a) 20% of marks
# An analytical solution for Eqs (2) is given by the following expressions for height $y$ and vertical speed $v_y$:
# 
# \begin{eqnarray*}
# y &=& y_0 - \frac{m}{k}\log_e\left[\cosh\left(\sqrt{\frac{kg}{m}}.t\right)\right]\quad(6)\\[2ex]
# v_y &=& -\sqrt{\frac{mg}{k}}\tanh\left(\sqrt{\frac{kg}{m}}.t\right)\quad(7)
# \end{eqnarray*}
# 
# These solutions apply for a free-falling object under constant gravity with constant drag factor, $k$.  In order to visualise them, adapt the code in the next cell.  Please note the following:
# 
# * You will need two arrays for the $y$ and $v_y$ values and you should plot them against your array of time values.
# 
# * You can set $y_0 = 1$ km, say, and calculate $y$ and $v_y$ for *any* $t$ using Eqs (6) and (7).  
# 
# * Choose sensible values for $k$ and $m$.
# 
# **N.B. You should make Python functions for the two expressions in Eqs (6) and (7), as these will be needed later in the exercise.**

# In[42]:


import numpy as np
import matplotlib.pyplot as plt

def height (tvals):
    """ 
    The height function is an analytical solution derived from Newton's second order 
    differential equation of motion for a free-falling object 
    under constant gravity with constant drag factor, k.
    """
    return (y0 - (m/k)*np.log(np.cosh(np.sqrt((k*g)/m)*tvals)))



def y_speed (tvals):
    """
    The vertical speed function is an analytical solution derived from Newton's second order
    differential equation of motion for a free-falling object
    under constant gravity with constant drag factor, k.
    """
    return (-(np.sqrt((m*g)/k)*np.tanh(np.sqrt((k*g)/m)*tvals)))


y0 = 1000  
m = 70 #about the mass of an adult male in kilograms
k = 0.3 #estimate for skydiver
g = 9.81 #accelerarion due to gravity at sea level (to 3sf.)

numpoints = 200
tmin = 0.0
tmax = 10.0
tvals = np.linspace(tmin,tmax,numpoints)  #200 points taken between t=0 and t=10s#
yvals = np.zeros(numpoints)
vyvals = np.zeros(numpoints) #taking positive as upwards so that falling velocity is -ve#

for i in range(numpoints):
    
    yvals[i] = height(tvals[i])
    vyvals[i] = y_speed(tvals[i])

plt.plot(tvals,yvals)
plt.xlabel('Time (s)')
plt.ylabel('Height (m)')
plt.show()
    
plt.plot(tvals,vyvals)
plt.xlabel('Time (s)')
plt.ylabel('Vertical Speed (m/s)')
plt.show()


# ---
# #### Part (b) 20% of marks
# Now solve the free-fall problem using the Euler method, as outlined in Eqs (3) to (5), noting the following:
# 
# * Using a starting height of 1 km and zero initial velocity, calculate $y(t)$ and $v_y(t)$ for the falling body. 
# * You will need to provide sensible values for $C_\mathrm{d}$, $A$ and $m$.   
# * You will also need to specify a condition for ending the simulation i.e. when the body reaches the ground. 
# * Put your time-stepping solution in the function template provided `freefall1()` - **this will be needed later in the exercise.**
# * The code provided will enable you to plot your results.  

# In[38]:


import sys
import numpy as np
import matplotlib.pyplot as plt

m = 70 #kg  average mass of male
g = 9.81 #m s^-2 acceleration due to gravity at sea level
Cd = 1.3 # drag coefficient for sky-diver
airdensity = 1.2 #kg m^-3 air density at ambient temperature and pressure
A = 0.3 #m^2 cross-sectional area of skydiver

k = (Cd*airdensity*A)/2 

t0 = 0
u0 = 0 # starting vertical speed
y0 = 1000 #m starting height

def freefall1 (y0, u0, maximumtime, numberofpoints, t, y, u):
   
    """
    Function to solve free fall equations using Euler method.
    """
    t[0] = t0
    y[0] = y0
    u[0] = u0
    
    dt = maximumtime/numberofpoints# time step
        
    for i in range(numberofpoints-1):
        
        t[i+1] = t[i] + dt # Eq 5
        u[i+1] = u[i] - dt*(g+((k/m)*abs(u[i])*u[i]))  # Eq 3
        
        if y[i] >= 0:
            y[i+1] = y[i] + dt*u[i] # Eq 4
        
        elif y[i] < 0:
            y[i] = np.nan
            u[i] = np.nan
            return
        
    finalstate = (t[numberofpoints-1],y[numberofpoints-1],u[numberofpoints-1])
    return finalstate


#===============================================================
# You need to set up your parameters here, and also create your
# numpy arrays.  You will need to estimate the number of data
# points and maximum time; or else set both of these very large
# to avoid running out of memory.
#===============================================================

maxtime = 30 # time in seconds. Extended from part a so to demonstrate end conditions
numpoints = 100 # number of points in simulation (including starting point)
startheight = 1000.0 # initial height in meters
startspeed = 0.0 # initial speed in m/s

t = np.linspace(0.0,maxtime,numpoints)
y = np.zeros(numpoints)
u = np.zeros(numpoints)

final_coords = freefall1(startheight,startspeed,maxtime,numpoints,t,y,u)

fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(12,4))
fig.suptitle('Height and speed data from Euler simulation of freefall')
ax1.set(xlabel='Time (s)', ylabel='Height (m)', title='Altitude')
ax2.set(xlabel='Time (s)', ylabel='Speed (m/s)', title='Vertical speed')

ax1.plot(t,y,'tab:red')
ax2.plot(t,u,'tab:green')
plt.show()


# ---
# #### Part (c) 20% of marks
# * In the cell below, you should call your `freefall1()` function a number of times with different values of $\Delta t$ and different vaues of the ratio $k/m$.  
# * You should plot your results, and include the analytical predictions on the same axes, using the functions you wrote in part (a).  
# * As a guide, you should produce plots for two different values of $\Delta t$ and two different values of $k/m$.
# * Make sure each plot is appropriately labelled with the parameters used.

# In[190]:


#===============================================================
# Call freefall1() here with different values of Delta t and
# different values of k/m.
#
# Using the code in part (b) as a guide, generate a set of plots
# with different delta-t and k/m values.
#===============================================================

t = np.linspace(0.0,maxtime,numpoints)
y = np.zeros(numpoints)
u = np.zeros(numpoints)


for numpoints in [50, 75, 100]:
    
    final_coords = freefall1(startheight,startspeed,maxtime,numpoints,t,y,u)
    
fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(12,4))
fig.suptitle('Height and speed data from Euler simulation of freefall')
ax1.set(xlabel='Time (s)', ylabel='Height (m)', title='Altitude')
ax2.set(xlabel='Time (s)', ylabel='Speed (m/s)', title='Vertical speed')

ax1.plot(t,y,'tab:red')
ax2.plot(t,u,'tab:green')
plt.show()


# In the cell below, briefly address the following points:
# 
# 1. How does the accuracy of your numerical solution vary with the step size used, $\Delta t$?
# 2. What is the largest value of $\Delta t$ that produces reliable solutions? Can you quantify the accuracy?
# 3. Describe the effect on the motion of varying the ratio $k/m$.

# [Your answers here]

# ---
# #### Part (d) 20% of marks
# Now you are going to make the problem more realistic.  Baumgartner jumped from very high altitude, where the air density is very low, and so the drag factor, $k$ should be replaced with a function $k(y)$.  The simplest way to approach this is to make use of the scale height for the atmosphere, $h$, and model the variation of density as an exponential decay:
# $$
# \rho(y) = \rho_0\exp(-y/h)
# $$
# 
# from which $k(y)$ follows using Eq. (1).  An appropriate value of $h$ appears to be 7.64 km [see http://en.wikipedia.org/wiki/Scale_height ].
# 
# * In the cell below, copy your `freefall1()` function to a new function `freefall2()`, and adapt it to use $\rho(y)$ instead of $\rho_0$.  
# 
# * Test your function using the parameters for Baumgartner's jump, and plot $y(t)$ and $v_y(t)$ against time.  

# In[34]:


def freefall2 (initial_height, initial_speed, maxtime, numpoints, tvalues, yvalues, vyvalues):
    """
    Function to solve free fall equations using Euler method, with variable air density.
    """

    #===========================================================
    # Make a new function freefall2() here using a variable
    # air density
    #===========================================================


# ---
# #### Part (e) 20% of marks
# There was great interest in whether Baumgartner would break the sound barrier during his jump. The speed of sound in a gas varies with the temperature:
# 
# $$v_s = \sqrt{\frac{\gamma RT}{M}}$$
# 
# where $\gamma$ is 1.4 for air, $R$ is the molar gas constant and $T$ is the absolute temperature in Kelvin. $M$ is the molar mass of the gas, which for dry air is about 0.028,964,5 kg/mol. The atmospheric temperature varies with altitude, $H$, as follows:
# 
# \begin{eqnarray*}
# \mbox{Troposphere:}\quad&H\leq11000\,\mathrm{m}\quad& T(\mathrm{K}) = 288.0-0.0065H\\
# \mbox{Lower Stratosphere:}\quad &11000<H\leq25100\,\mathrm{m}\quad& T(\mathrm{K}) = 216.5\\
# \mbox{Upper Stratosphere:}\quad& H>25100\,\mathrm{m}\quad& T(\mathrm{K}) = 141.3+0.0030H
# \end{eqnarray*}
# 
# Using the information above and your function `freefall2()`, write a short program in the following cell to determine Baumgartner's maximum Mach number (fraction of the speed of sound) as he falls.
# 

# In[ ]:


#===========================================================
# Write your program here
#===========================================================


# ---
# ## Submitting your work
# Submit your completed Jupyter notebook, bearing in mind the following points:
# 
# * Blackboard plagiarism checker (Turnitin) won't accept a file ending `.ipynb` so please **rename** your file with a `.txt` extension.[1] 
# 
# * Do **not** use the *cut-and-paste* submission option as this will not understand the markdown language used in this notebook and will likely make the file unusable. 
# 
# * Please give your notebook a sensible distinguishing name, including your name or userid e.g. `my_userid_ex2.txt`.  
# 
# * If you have any problems submitting your work, please contact Dr. Hanna (s.hanna@bristol.ac.uk) or ask a demonstrator.
# 
# [1] Strictly speaking, Turnitin can be instructed to accept files with the `.ipynb` extension.  However, it will only apply plagiarism checking if the `.txt` extension is used.

# In[ ]:




