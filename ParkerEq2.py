import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.constants import G, k, m_p

#Constants
M_sun = 1.9891e30 #kg   
R_sun = 6.96e8 #m          
T = 1e6 #K         
mu = 1                        
cs = np.sqrt(k*T/(mu*m_p)) #m/s
rc = G*M_sun/(2*cs**2) #m

#Boundary Conditions
r0 = 1.01*rc #m    
rho0 = 1e8*m_p #kg/m^3 Density boundary condition at r0, can be changed
v0 = 1.01*cs

#Integrate around critical point 
eps = 1e-4
r1 = rc * (1 - eps) #just behind critical point
r2 = rc * (1 + eps) #just in front of critical point
v1 = cs * (1 - eps) #boundary velocity just smaller than sound speed
v2 = cs * (1 + eps) #boundary velocity just larger than sound speed

r_min = R_sun
r_max = 1e3 * R_sun
r_eval1 = np.linspace(r_min, r1, 1000)
r_eval2 = np.linspace(r2, r_max, 1000)

#Equation for dv/dr to pass to solver
def Parker(r, v):
    numerator = v*(2*cs**2)*(1-rc/r)
    denominator =  r*(v**2 - cs**2)
    result = numerator/denominator
    return [result]

#Generate r values and integration range
r_max = 100*rc
r_vals = np.linspace(r0, r_max, 1500)

#Check inputs
print(f'r_c = {rc/R_sun} solar radii')
print(f'r_min = {r0/R_sun} solar radii')
print(f'r_max = {r_max/R_sun} solar radii')
print(f'Sound speed = {cs/1e3} km/s')
print(f'v(r0) = {v0/1e3} km/s')

#Try solver 
try:
    sol = solve_ivp(Parker, [r0, r_max], [v0], t_eval=r_vals, method='RK45')
    if sol.success:
        print("Solver was successful.")
        r_sol = sol.t
        v_sol = sol.y[0]
        rho_sol = rho0*((r0/r_sol)**2)*(v0/v_sol) #Density equation based on boundary conditions 
        r_sol_R_sun = r_sol/R_sun #r in units of solar radii
        v_sup = 2*cs*np.sqrt(np.log(r_sol/rc)) #Analytic approximation
        
        valid = (r_sol > rc)

        plt.figure(figsize=(12, 5))
        plt.suptitle(f'Isothermal Solar Wind at T = {T:.2e} K')

        #Velocity
        plt.subplot(1, 2, 1)
        plt.plot(r_sol_R_sun, v_sol/1e3, label = r'$v(r)$')
        plt.plot(r_sol_R_sun[valid], v_sup[valid]/1e3, color = 'k', linestyle = '--', label = r'Supersonic limit, $v(r) \gg c_s$')
        plt.xlabel(r'Radial Distance $(R_{\bigodot})$')
        plt.ylabel(r'Velocity $(km \: s^{-1})$')
        # plt.xscale('log')
        plt.title('Wind Velocity Profile')
        plt.legend()
        plt.grid(True)

        #Density
        plt.subplot(1, 2, 2)
        plt.plot(r_sol_R_sun, rho_sol / m_p / 1e6)
        plt.xlabel(r'Radial Distance $(R_{\bigodot})$')
        plt.ylabel(r'Density $(cm^{-3}$)')
        plt.title('Wind Density Profile')
        plt.yscale('log')
        plt.grid(True)

        plt.tight_layout()
        plt.show()  

    else:
        print("Solver was unsuccessful.")
except Exception as e:
    print(f"Error during integration: {e}")




