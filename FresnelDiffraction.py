# -*- coding: utf-8 -*-

# Level 5 Laboratory - Computational Physics

# Bertie Thorpe 22/02/21

# Fresnel Diffraction - Simpson's Integration

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Part A - Simpson's integration of 1-dimensional Fresnel diffraction integral

epsilon0 = 8.85418782e-12
c = 2.99792458e+8
wavelength = 1e-6
k = (2*np.pi)/wavelength
z = 2e-5
x = -1e-2
N = 100
ax_a = -2e-5
ax_b = -ax_a
aperture_x = ax_b - ax_a

def FresnelExp(x,aperture_x,z,k):
    '''
    The exponential integrand of the 1d Fresnel integral function.
    '''
    
    return np.exp(0.5j*k*((x-aperture_x)**2)/z)

def Simpson1d(FresnelExp,x,ax_a,N,z):
    ''' 
    Approximate the integral of f(x) from x = a to x = b by Simpson's rule.
    with N subintervals of equal length. N must be even integer.
    '''
    
    if N % 2 != 0:  # N must be even
        return None
    
    h = (ax_b-ax_a)/N
    first = FresnelExp(x,ax_a,z,k)
    last = FresnelExp(x,ax_b,z,k)
    
    aperture_x = ax_a
    s = 0
    
    for i in np.arange(N-1):
        aperture_x += h
        value = FresnelExp(x,aperture_x,z,k)
        
        if i % 2 == 0:
            s += 2 * value
        else:
            s += 4 * value
        
    integral = (h/3)*(first+s+last)
    electricfield = np.abs((k/(2*z*np.pi))*integral)
    return electricfield

electricfield = Simpson1d(FresnelExp,x,ax_a,N,z)
relativeintensity = epsilon0*c*(Simpson1d(FresnelExp,x,ax_a,N,z))**2
NumPoints = 200
xmin = -1e-2
xmax = +1e-2
dx = (xmax - xmin) / (NumPoints - 1)  # i start counting from 0

xvals = np.zeros(NumPoints)
yvals = np.zeros(NumPoints)

for i in np.arange(NumPoints):
    xvals[i] = xmin + i * dx
    yvals[i] = epsilon0*c*(Simpson1d(FresnelExp,xvals[i],ax_a,N,z))**2

fig1 = plt.figure(1)
plotcolor = 'lightgoldenrodyellow'
ax = fig1.subplots()
plt.yscale('symlog',linthresh=1000)
plt.subplots_adjust(bottom=0.3)
p, = ax.plot(xvals,yvals,'firebrick')

z_slide = plt.axes([0.25,0.15,0.61,0.03], facecolor=plotcolor)
aperture_slide = plt.axes([0.25,0.1,0.61,0.03], facecolor=plotcolor)
N_slide = plt.axes([0.25,0.05,0.61,0.03], facecolor=plotcolor)

z_factor = Slider(z_slide,'Screen Distance (m)',valmin=0.00002,valmax=0.02,valinit=0.00002,valstep=0.00002)
aperture_factor = Slider(aperture_slide,'Aperture Width (m)',valmin=5e-6,valmax=5e-5,valinit=4e-5,valstep=1e-6)
N_factor = Slider(N_slide,'Integration Intervals',valmin=50,valmax=400,valinit=100,valstep=2)

def update1(val):
    current_z_val = z_factor.val
    current_aperture_val = (aperture_factor.val)/2
    current_N_val = N_factor.val
    p.set_ydata(epsilon0*c*(Simpson1d(FresnelExp,xvals,ax_a=-current_aperture_val,N=current_N_val,z=current_z_val)**2))
    fig1.canvas.draw() # redraw the figure
    ax.relim()
    ax.autoscale_view()
    
z_factor.on_changed(update1)
aperture_factor.on_changed(update1)
N_factor.on_changed(update1)

ax.set_facecolor(plotcolor)
ax.set_xlabel('Screen Position (m)', fontsize=10)
ax.set_ylabel('Relative Intensity', fontsize=10)
 
def Fresnel2d():
    
    current_z_val = z_factor.val
    current_aperture_val = (aperture_factor.val)/2
    current_N_val = N_factor.val
    
    intensity = np.zeros((NumPoints,NumPoints))
    
    for l in np.arange(NumPoints):
        xvals2 = xmin + l * dx
        
        for p in np.arange(NumPoints):
                yvals2 = xmin + (p * dx)
                intensity[l,p] = epsilon0*c*(((np.abs(Simpson1d(FresnelExp, xvals2, ax_a=-current_aperture_val, N=current_N_val, z=current_z_val)))*Simpson1d(FresnelExp, yvals2, ax_a=-current_aperture_val, N=current_N_val, z=current_z_val))/(k/(2*(current_z_val)*np.pi)))**2
        
    plt.figure(2)
    plt.imshow(intensity)
    plt.axis('off')
    cbar = plt.colorbar()
    cbar.set_label('Relative Intensity')
    
d={1:Fresnel2d}

print('Adjust parameters in 1d window, THEN type "d[1]()" to run Fresnel2d function. Warning: A lot integration intervals makes the program run very slow')
plt.show()









