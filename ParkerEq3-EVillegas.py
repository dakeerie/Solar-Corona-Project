import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.constants import G, k, m_p
from sympy import *
from scipy.optimize import root_scalar

#Constants
M_sun = 1.9891e30 #kg   
R_sun = 6.96e8 #m          
T = 1e6 #K         
mu = 1                        
cs = np.sqrt(2*k*T/(mu*m_p)) #m/s
rc = G*M_sun/(2*cs**2) #m

#Dimensionless ODE
def Parker_ode(R, v):
    numerator  = 2*(1/R - 1/R**2)
    denominator = (v-1/v)
    result = numerator/denominator
    return result

def RK4(dydx, initial_conditions, x_final, n):
    # RK4 solver
    x = [initial_conditions[0]]
    y = [initial_conditions[1]]
    dx = (x_final - x[0])/(n - 1) # step
    dx_2 = dx/2 # half step
    for i in range(n-1):
        k1 = dydx(x[i], y[i])
        k2 = dydx(x[i] + dx_2, y[i] + dx*k1/2)
        k3 = dydx(x[i] + dx_2, y[i] + dx*k2/2)
        x.append(x[i] + dx) # x[i+1] = x[-1]
        k4 = dydx(x[-1], y[i] + dx*k3)
        y.append(y[i] + dx*(k1 + 2*k2 + 2*k3 + k4)/6)
    return x, y

#Integration range
r_min = R_sun #m
r_max = 3e2*R_sun #m
R_min = r_min/rc #dimensionless
R_max = r_max/rc #dimensionless
eps = 1e-7

#Boundary Condition
n0 = 5e6 #cm^-3
rho0 = n0*m_p*10e6 #kg/m^3

#Check inputs
print()
print(f'Inputs:')
print(f'r_c = {rc/R_sun} solar radii')
print(f'r_min = {r_min/R_sun} solar radii')
print(f'r_max = {r_max/R_sun} solar radii')
print(f'Sound speed = {cs/1e3} km/s')
print()
print('Boundary Condition:')
print(f'rho(r0) = {rho0} kg/cm^3')


#Solve
#Interior solution
R_interior, v_interior = RK4(Parker_ode, [1, 1 - eps], R_min, 50000)
#This part needs reversed as above RK4 integrates from the critical point back in r to R_min
R_interior.reverse()
v_interior.reverse()
R_exterior, v_exterior = RK4(Parker_ode, [1, 1 + eps], R_max, 600000)

R_solution = R_interior + R_exterior #dimensionless solution
R_solution = np.array(R_solution)
V_solution = v_interior + v_exterior #dimensionless solution
V_solution = np.array(V_solution)

#Add dimensions back
r_solution = rc*R_solution
v_solution = cs*V_solution

plt.figure()
plt.plot(R_solution, V_solution, label = 'V(R)')
plt.plot(1, 1, 'or', label = 'Critical point')
plt.title('Dimensionless Velocity Profile', fontsize = 20)
plt.xlabel(r'$R = \frac{r}{r_c}$', fontsize = 20)
plt.ylabel(r'$V = \frac{v}{c_s}$', fontsize = 20)
plt.grid(True, which = 'both')
plt.legend()
# plt.show()


plt.figure()
plt.plot(r_solution/R_sun, v_solution/1e3, label = 'v(r)')
plt.plot(rc/R_sun, cs/1e3, 'or', label = 'Critical point')
plt.plot(r_solution/R_sun, np.sqrt(4*cs**2*np.log(r_solution/rc) - 3)/1e3, label = 'Analytical supersonic limit')
plt.title('Velocity profile', fontsize = 20)
plt.xlabel(r'$\frac{r}{R_{\bigodot}}$', fontsize = 20)
plt.ylabel('v (km/s)', fontsize = 20)
plt.grid(True, which = 'both')
plt.legend()
# plt.show()

#Density profile
index = np.argmin(np.abs(r_solution - 2*R_sun))
const = rho0*v_solution[index]*r_solution[index]**2
rho_solution = const/(v_solution*r_solution**2)
rc_idx = np.argmin(np.abs(r_solution - rc))
v_analytical = 2*cs*np.sqrt(np.log(r_solution/rc))


plt.figure()
plt.plot(r_solution/R_sun, rho_solution, label = 'Density')
plt.plot(rc/R_sun, rho_solution[rc_idx], '.', label = 'Critical point')
plt.plot(r_solution/R_sun, const/(v_analytical*r_solution**2), label = 'Analytical approximation')
plt.xlabel(r'$\log{\frac{r}{R_{\bigodot}}}$', fontsize = 20)
plt.ylabel(r'$\log{\rho (r)}$', fontsize = 20)
plt.title('Density Profile', fontsize = 20)
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.grid()
# plt.show()

M_dot = rho_solution*v_solution*r_solution**2
#Maximum variation of mass loss rate
variation = (np.max(M_dot) - np.min(M_dot))/(np.mean(M_dot))*100
print(f'Maximum variation is {variation} %')

plt.figure()
plt.plot(r_solution/R_sun, M_dot, label = 'Mass loss rate')
plt.plot(rc/R_sun, M_dot[rc_idx], '.', label = 'Critical point')
plt.xlabel(r'$\frac{r}{R_{\bigodot}}$', fontsize = 20)
plt.ylabel(r'$\dot{M} (kg/s)$', fontsize = 20)
plt.title('Mass loss rate', fontsize = 20)
plt.legend()
plt.grid()
plt.show()





# R_ext1, v_ext1 = RK4(Parker_ode, [1, 1+1e-6], 3, 20000)
# R_ext2, v_ext2 = RK4(Parker_ode, [3, v_ext1[-1]], R_max, 10000)
# R_int.extend(R_ext1)
# v_int.extend(v_ext1)
# R_int.extend(R_ext2)
# v_int.extend(v_ext2)
# R_sol = R_int #dimensionless solutions
# V_sol = v_int #dimensionless solutions
# #Add dimensions
# r_sol = np.array(R_sol)*rc #m
# r_sol_R_sun = r_sol/R_sun #solar radii
# v_sol = np.array(V_sol)*cs #m/s


# #Density profile
# const = rho0*v_sol[0]*r_sol[0]**2 #kg/s
# rho_sol = (const)/(v_sol*r_sol**2) #kg/m^3

# mdot0 = 4*np.pi*r_sol[0]**2*v_sol[0]*rho0
# print('Mdot0 = ' + '{0:.3f}'.format(mdot0) + ' kg/s')

# #Analytical approximations
# supersonic = (r_sol > rc)
# subsonic = (r_sol < rc)
# v_sup = 2*cs*np.sqrt(np.log(r_sol[supersonic]/rc)) #m/s
# rho_sup = (const)/(2*m_p*cs*(r_sol[supersonic]**2)*np.sqrt(np.log(r_sol[supersonic]/rc)))
# n_full = 4.8e9*(R_sun/r_sol)**14 + 3e8*(R_sun/r_sol)**6 + 1.4e6*(R_sun/r_sol)**2.3 #Analytic density approximation from Kontar et al. 2023
# n_subsonic = 5.14e9*np.exp((G*M_sun*m_p/(k*T*R_sun))*(R_sun/r_sol - 1)) #Natural log of density
# # exponent = ((G*M_sun/k/T)/R_sun) * (R_sun/r_sol - 1)
# print(n_subsonic)


# # x - ln(x) = 4ln(r/rc) +4rc/r - 3

# # def solve_v(r):
# #     rhs = 4*np.log(r/rc) + 4*(rc/r) - 3
# #     # function in x = (v/v_c)^2
# #     def f(x):
# #         return x - np.log(x) - rhs
# #     # initial guess: x ~ 1
# #     sol = root_scalar(f, bracket=[1e-6, 50], method='bisect')
# #     x = sol.root
# #     return cs * np.sqrt(x)

# # v_vals = [solve_v(r) for r in r_sol]

# # plt.figure(figsize = (6,5))
# # plt.plot(r_sol_R_sun[0:100] - 1, v_sol[0:100]/1e3, label = r'$v(r)$')
# # plt.xlabel(r'$\frac{r}{R_{\bigodot}}$', fontsize = 15)
# # plt.ylabel(r'Velocity $(km \: s^{-1})$', fontsize = 10)
# # plt.title('Wind Velocity Profile for small r')
# # plt.xlabel(r'$\frac{r}{R_{\bigodot}} - 1$', fontsize = 15)
# # plt.ylabel(r'Velocity $(km \: s^{-1})$', fontsize = 10)
# # plt.grid(True, which = 'both')
# # plt.xscale('log')
# # plt.yscale('log')
# # plt.legend(loc = 'upper right')
# # plt.show()

# plt.figure(figsize=(14, 6))
# plt.suptitle(f'Isothermal Solar Wind at T = {T:.2e} K')

# #Velocity Plot
# plt.subplot(1, 2, 1)
# plt.plot(r_sol_R_sun - 1, v_sol/1e3, label = r'$v(r)$')
# plt.plot(r_sol_R_sun[supersonic], v_sup/1e3, color = 'k', linestyle = '--', label = r'Analytic approximation, $v(r) \gg c_s$')
# plt.xlabel(r'$\frac{r}{R_{\bigodot}} - 1$', fontsize = 15)
# plt.ylabel(r'Velocity $(km \: s^{-1})$', fontsize = 10)
# plt.title('Wind Velocity Profile')
# plt.axvline(rc/R_sun, ymin = 0, ymax = 1, linestyle = '--', color = 'magenta', label = 'Critical radius and speed')
# plt.axhline(cs/1e3, xmin = 0, xmax = 1, linestyle = '--', color = 'magenta')
# plt.grid(True, which = 'both')
# plt.xscale('log')
# plt.yscale('log')
# plt.legend(loc = 'best')

# #Number Density Plot
# plt.subplot(1, 2, 2)
# plt.plot(r_sol_R_sun - 1, rho_sol/m_p, label = r'$\rho(r)$')
# plt.plot(r_sol_R_sun[supersonic] - 1, rho_sup, color='k', linestyle='--', label=r'$\rho \propto \frac{1}{r^2 \sqrt{\ln{r}}}$')
# plt.plot(r_sol_R_sun -1 , n_full, linestyle = '--', color = 'red', label = r'Eq. 11 from Kontar et al. 2023')
# plt.plot(r_sol_R_sun -1, n_subsonic, linestyle = '--', color = 'green', label = r'Eq. 18 Kontar 2019')
# plt.xlabel(r'$\frac{r}{R_{\bigodot}} - 1$', fontsize = 15)
# plt.ylabel(r'Density $(cm^{-3}$)', fontsize = 10)
# plt.axvline(rc/R_sun, ymin = 0, ymax = 1, linestyle = '--', color = 'magenta', label = 'Critical radius')
# plt.title('Wind Density Profile')
# plt.yscale('log')
# plt.xscale('log')
# plt.legend(loc = 'best')
# plt.grid(True, which = 'both')
# plt.show()

# plt.figure(figsize=(12,5))
# plt.plot(r_sol_R_sun-1, 4*np.pi*r_sol**2*rho_sol*v_sol)
# plt.title(f'Mass loss rate for isothermal wind at T = {T:.2e} K')
# plt.xlabel(r'$\frac{r}{R_{\bigodot}}$ - 1', fontsize = 15)
# plt.ylabel(r'$\dot{M} \: (kg\: s^{-1})$')
# # plt.yscale('log')
# plt.tight_layout()
# plt.show() 


