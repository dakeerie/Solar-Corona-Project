import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.constants import G, k, m_p

#Constants
M_sun = 1.9891e20 #kg   
R_sun = 696300e3 #m          
T = 1e6 #K         
mu = 1                        
cs = np.sqrt(k*T/(mu*m_p)) #m/s
rc = G*M_sun/(2*cs**2) #m

#Boundary Conditions
r0 = 1.01*rc #m    
rho0 = 1e8*m_p #Density boundary condition at r0, can be changed
U0 = 1.01*cs

#Equation for dv/dr to pass to solver
def Parker(r, v):
    numerator = v*(2*cs**2)*(1-rc/r)
    denominator =  v**2 - cs**2
    result = numerator/denominator
    return [result]

#Generate r values and integration range
r_max = 30*rc
r_vals = np.linspace(r0, r_max, 1000)

#Try solver 
try:
    sol = solve_ivp(Parker, [r0, r_max], [U0], t_eval=r_vals, method='RK45')
    if sol.success:
        print("Successful.")
        r_sol = sol.t
        v_sol = sol.y[0]
        rho_sol = rho0*((r0/r_sol)**2)*(U0/v_sol) #Density equation based on boundary conditions 
        r_sol_R_sun = r_sol/R_sun #r in units of solar radii

        plt.figure(figsize=(12, 5))

        #Velocity
        plt.subplot(1, 2, 1)
        plt.plot(r_sol_R_sun, v_sol / 1e3)
        plt.xlabel('Radial Distance [Solar Radii]')
        plt.ylabel('Velocity [km/s]')
        plt.title('Solar Wind Speed Profile')
        plt.grid(True)

        #Density
        plt.subplot(1, 2, 2)
        plt.plot(r_sol_R_sun, rho_sol / m_p / 1e6)
        plt.xlabel('Radial Distance [Solar Radii]')
        plt.ylabel('Density [cm⁻³]')
        plt.title('Solar Wind Density Profile')
        plt.yscale('log')
        plt.grid(True)

        plt.tight_layout()
        plt.show()  

    else:
        print("Unsuccessful.")
except Exception as e:
    print(f"Error during integration: {e}")




