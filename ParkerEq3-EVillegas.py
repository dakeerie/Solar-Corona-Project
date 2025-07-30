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

#Boundary Conditions
eps = 1e-4
r0 = R_sun #m
v0 = 1000 #m/s
rho0 = 1e8*m_p #kg cm^-3

#Check inputs
print()
print(f'Inputs:')
print(f'r_c = {rc/R_sun} solar radii')
print(f'r_min = {r_min/R_sun} solar radii')
print(f'r_max = {r_max/R_sun} solar radii')
print(f'Sound speed = {cs/1e3} km/s')
print()
print('Boundary Conditions:')
print(f'r0 = {r0/R_sun} solar radii')
print(f'v(r0) = {v0/1e3} km/s')
print(f'rho(r0) = {rho0} kg/cm^3')

#Solve
R_int, v_int = RK4(Parker_ode, [1, 1-1e-6], R_min, 10000)
R_int.reverse()
v_int.reverse()
R_ext1, v_ext1 = RK4(Parker_ode, [1, 1+1e-6], 3, 20000)
R_ext2, v_ext2 = RK4(Parker_ode, [3, v_ext1[-1]], R_max, 10000)
R_int.extend(R_ext1)
v_int.extend(v_ext1)
R_int.extend(R_ext2)
v_int.extend(v_ext2)
R_sol = R_int #dimensionless solutions
V_sol = v_int #dimensionless solutions
#Add dimensions
r_sol = np.array(R_sol)*rc #m
r_sol_R_sun = r_sol/R_sun #solar radii
v_sol = np.array(V_sol)*cs #m/s
#Density profile
const = rho0*v0*r0**2 #kg/s
rho_sol = (const)/(v_sol*r_sol**2) #kg/m^3

#Analytical approximations
valid = (r_sol > rc)
v_sup = 2*cs*np.sqrt(np.log(r_sol[valid]/rc)) #m/s
rho_sup = (const)/(2*m_p*cs*(r_sol[valid]**2)*np.sqrt(np.log(r_sol[valid]/rc)))
n = 4.8e9*(R_sun/r_sol)**14 + 3e8*(R_sun/r_sol)**6 + 1.4e6*(R_sun/r_sol)**2.3 #Analytic density approximation from Kontar et al. 2023

plt.figure(figsize = (6,5))
plt.plot(r_sol_R_sun[0:100], v_sol[0:100]/1e3, label = r'$v(r)$')
plt.xlabel(r'$\frac{r}{R_{\bigodot}}$', fontsize = 15)
plt.ylabel(r'Velocity $(km \: s^{-1})$', fontsize = 10)
plt.title('Wind Velocity Profile')
plt.grid(True, which = 'both')
plt.xscale('log')
plt.legend(loc = 'upper right')
plt.show()

plt.figure(figsize=(12, 5))
plt.suptitle(f'Isothermal Solar Wind at T = {T:.2e} K')

#Velocity Plot
plt.subplot(1, 2, 1)
plt.plot(r_sol_R_sun, v_sol/1e3, label = r'$v(r)$')
plt.plot(r_sol_R_sun[valid], v_sup/1e3, color = 'k', linestyle = '--', label = r'Analytic approximation, $v(r) \gg c_s$')
plt.xlabel(r'$\frac{r}{R_{\bigodot}}$', fontsize = 15)
plt.ylabel(r'Velocity $(km \: s^{-1})$', fontsize = 10)
plt.title('Wind Velocity Profile')
plt.plot(rc/R_sun, cs/1e3, 'x', color = 'magenta', label = 'Critical point' )
plt.grid(True, which = 'both')
plt.xscale('log')
plt.legend(loc = 'upper right')

#Number Density Plot
plt.subplot(1, 2, 2)
plt.plot(r_sol_R_sun, rho_sol/m_p, label = r'$\rho(r)$')
plt.plot(r_sol_R_sun[valid], rho_sup, color='k', linestyle='--', label=r'$\rho \propto \frac{1}{r^2 \sqrt{\ln{r}}}$')
plt.plot(r_sol_R_sun, n, linestyle = '--', color = 'red', label = r'Eq. 11 from Kontar et al. 2023')
plt.xlabel(r'$\frac{r}{R_{\bigodot}}$', fontsize = 15)
plt.ylabel(r'Density $(cm^{-3}$)', fontsize = 10)
plt.plot(r0/R_sun, rho0/m_p, 'x', color = 'magenta', label = 'Boundary Condition')
plt.title('Wind Density Profile')
plt.yscale('log')
plt.xscale('log')
plt.legend(loc = 'upper right')
plt.grid(True, which = 'both')

plt.tight_layout()
plt.show() 

