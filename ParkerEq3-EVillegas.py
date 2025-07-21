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
    numerator  = 2*(1/R - 1/(R**2))
    denominator = (v - 1/v)
    result = numerator/denominator
    return result

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
sol_int = solve_ivp(Parker_ode, [R_min, 1], [1 - 1e-6], method = 'RK45')
sol_ext = solve_ivp(Parker_ode, [1, R_max], [1 + 1e-6], method = 'RK45')
#Extract solutions
R_int = sol_int.t
v_int = sol_int.y[0]
R_ext = sol_ext.t
v_ext = sol_ext.y[0]
#Combine
R_sol = np.concatenate((R_int, R_ext))
v_sol = np.concatenate((v_int, v_ext))

plt.plot(R_sol, v_sol)
plt.xscale('log')
plt.yscale('log')
plt.show()


