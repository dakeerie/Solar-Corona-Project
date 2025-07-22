import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.constants import G, k, m_p
from sympy import *

#NUMERICAL APPROACH

#Constants
M_sun = 1.9891e30 #kg   
R_sun = 6.96e8 #m          
T = 1e6 #K         
mu = 1                        
cs = np.sqrt(k*T/(mu*m_p)) #m/s
rc = G*M_sun/(2*cs**2) #m
#Include solar rotation rate??

#Dimensionless ODE function
def Parker_ode(R, v):
    numerator  = 2*(1/R - 1/R**2)
    denominator = (v-1/v)
    result = numerator/denominator
    return result

def RK4(function, initial_conditions, x_final, n):
    # RK4 solver
    x = [initial_conditions[0]]
    w = [initial_conditions[1]]
    dx = (x_final - x[0]) / n # step
    dx_2 = dx / 2 # half step
    for i in range(n-1):
        k1 = dx * function(x[i], w[i])
        k2 = dx * function(x[i] + dx_2, w[i] + k1/2)
        k3 = dx * function(x[i] + dx_2, w[i] + k2/2)
        x.append(x[i] + dx) # x[i+1] = x[-1]
        k4 = dx * function(x[-1], w[i] + k3)
        w.append(w[i] + (k1 + 2*k2 + 2*k3 + k4)/6)
    return x, w

#Integration range
r_min = R_sun #m
r_max = 1e3*R_sun #m
R_min = r_min/rc #dimensionless
R_max = r_max/rc #dimensionless

#Check inputs
print(f'r_c = {rc/R_sun} solar radii')
print(f'r_min = {r_min/R_sun} solar radii')
print(f'r_max = {r_max/R_sun} solar radii')
print(f'Sound speed = {cs/1e3} km/s')
# print(f'v(r0) = {v0/1e3} km/s')

#Solve
R_int, v_int = RK4(Parker_ode, [1, 1-1e-6], R_min, 300000)
R_int.reverse()
v_int.reverse()
R_ext, v_ext = RK4(Parker_ode, [1, 1+1e-6], R_max, 600000)
R_int.extend(R_ext)
v_int.extend(v_ext)
R_sol = R_int #dimensionless solutions
v_sol = v_int #dimensionless solutions

plt.plot(R_sol, v_sol)
plt.xscale('log')
# plt.yscale('log')
plt.show()


