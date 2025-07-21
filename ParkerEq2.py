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

#Integrate around critical point 
eps = 1e-3
r1 = rc * (1 - eps) #just behind critical point
r2 = rc * (1 + eps) #just in front of critical point
v1 = cs * (1 - eps) #boundary velocity just smaller than sound speed
v2 = cs * (1 + eps) #boundary velocity just larger than sound speed

#Generate r values and integration range
r_min = R_sun
r_max = 1e3 * R_sun
r_eval1 = np.linspace(r_min, r1, 1000)
r_eval2 = np.linspace(r2, r_max, 1000)

#Boundary Conditions
#These BCs are past the critical radius!!
r0 = r2 #m     
rho0 = 1e8*m_p #kg/cm^3 Density boundary condition at r0, can be changed
v0 = v2 #m/s

#Equation for dv/dr to pass to solver
def Parker(r, v):
    numerator = v*(2*cs**2)*(1-rc/r)
    denominator =  r*(v**2 - cs**2)
    result = numerator/denominator
    return [result]

#Check inputs
print(f'r_c = {rc/R_sun} solar radii')
print(f'r_min = {r_min/R_sun} solar radii')
print(f'r_max = {r_max/R_sun} solar radii')
print(f'Sound speed = {cs/1e3} km/s')
print(f'v(r0) = {v0/1e3} km/s')

#Try solver 
try:
    sol_outer = solve_ivp(Parker, [r2, r_max], [v2], t_eval=r_eval2, method = 'RK45')
    # sol_inner = solve_ivp(Parker, [r_min, r1], [v1], t_eval=r_eval1, method = 'BDF', rtol = 1e-9, atol = 1e-12)
    if sol_outer.success:
        print("Both solutions were successful.")
        # r_sol = np.concatenate([sol_inner.t, [rc], sol_outer.t])
        # v_sol = np.concatenate([sol_inner.y[0], [cs], sol_outer.y[0]])
        r_sol = sol_outer.t
        v_sol = sol_outer.y[0]
        const = rho0*v0*r0**2
        rho_sol = const/(v_sol*r_sol**2) #Density equation based on boundary conditions 
        r_sol_R_sun = r_sol/R_sun #r in units of solar radii
        v_sup = 2*cs*np.sqrt(np.log(r_sol/rc)) #Analytic velocity approximation 
        # n = 4.8e9*(R_sun/r_sol)**14 + 3e8*(R_sun/r_sol)**6 + 1.4e6*(R_sun/r_sol)**2.3 #Analytic density approximation from Kontar et al. 2023
        
        valid = (r_sol > rc)

        plt.figure(figsize=(12, 5))
        plt.suptitle(f'Isothermal Solar Wind at T = {T:.2e} K')

        #Velocity
        plt.subplot(1, 2, 1)
        plt.plot(r_sol_R_sun, v_sol/1e3, label = r'$v(r)$')
        plt.plot(r_sol_R_sun[valid], v_sup[valid]/1e3, color = 'k', linestyle = ':', label = r'Analytic approximation, $v(r) \gg c_s$')
        plt.axvline(rc/R_sun, linestyle = '--', color = 'black', label = r'$r_c$ and $c_s$ lines' )
        plt.axhline(cs/1e3, linestyle = '--', color = 'black')
        plt.xlabel(r'$\frac{r}{R_{\bigodot}}$', fontsize = 15)
        plt.ylabel(r'Velocity $(km \: s^{-1})$', fontsize = 10)
        plt.xscale('log')
        # plt.yscale('log')
        plt.title('Wind Velocity Profile')
        plt.legend()
        plt.grid(True, which = 'both')

        #Density
        plt.subplot(1, 2, 2)
        plt.plot(r_sol_R_sun, rho_sol/(m_p) , label = r'$\rho(r)$')
        plt.plot(r_sol_R_sun, (const/m_p)/(2*cs*(r_sol**2)*np.sqrt(np.log(r_sol/rc))), color='k', linestyle=':', label=r'Analytic approximation, $\rho \propto \frac{1}{r^2 \sqrt{\ln{r}}}$')
        # plt.plot(r_sol_R_sun, n, linestyle = ':', color = 'red', label = r'Analytic approximation, eq. 11 from Kontar et al. 2023')
        plt.axvline(rc/R_sun, linestyle = '--', color = 'k', label = r'$r_c$ line')
        plt.xlabel(r'$\frac{r}{R_{\bigodot}}$', fontsize = 15) #r in metres FOR NOW!!!
        plt.ylabel(r'Density $(cm^{-3}$)', fontsize = 10)
        plt.title('Wind Density Profile')
        plt.yscale('log')
        plt.xscale('log')
        plt.legend()
        plt.grid(True, which = 'both')

        plt.tight_layout()
        plt.show()  

    else:
        print("Solver was unsuccessful.")
except Exception as e:
    print(f"Error during integration- {e}")




