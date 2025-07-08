import numpy as np
import scipy.constants as const
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt

def Parker(r, U, cs, rc):
    numerator = ((2*U*cs**2)/r)*(1-rc/r)
    denominator = (U**2 - cs**2)
    eps = 1e-12
    if abs(denominator) < eps:
        denominator = eps if denominator >= 0 else -eps
    dUdr = numerator/denominator
    max_deriv = 1e5
    dUdr = np.clip(dUdr, -max_deriv, max_deriv)
    return [dUdr]  # Return as list

AU = 1.496e11 #m
r_min = 0.2*AU
r_max = 1.1*AU
r_eval = np.linspace(r_min, r_max, 30)

M_sun = 1.989e30 #kg
T0 = 1e6
mu0 = 0.6
cs = np.sqrt(2*const.k*T0/(mu0*const.m_p))
U0 = 1.3*cs
rc = const.G*M_sun/(2*cs)

try:
    sol = solve_ivp(Parker, [r_min, r_max], [U0], t_eval=r_eval, args=(cs, rc), method='BDF', rtol=1e-4, atol = 1e-6)
    if sol.success:
        print("Solution successful.")
        r_solution = sol.t/AU
        U_solution = sol.y[0]
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(r_solution, U_solution/1000, 'b-', linewidth=2, label='U(r)')
        ax1.set_xlabel('Distance (AU)')
        ax1.set_ylabel('Velocity (km/s)')
        ax1.set_title('Solar Wind Velocity Profile')
        ax1.grid(True)
        ax1.axhline(y=cs/1000, color='r', linestyle='--', label=f'Sound speed ({cs/1000:.1f} km/s)')
        ax1.legend()
        plt.tight_layout()
        plt.show()

        # ax2.semilogy(r_solution, density, 'r-', linewidth=2)
        # ax2.semilogy(r_solution, 1/((r_solution**2)*np.sqrt(np.log(r_solution))), label = 'Density profile proportionality')
        # ax2.set_xlabel('Distance (AU)')
        # ax2.set_ylabel('Density (particles/m³)')
        # ax2.set_title('Solar Wind Density Profile')
        # ax2.grid(True)

        idx_1au = np.argmin(np.abs(r_solution - 1.0))
        print(f"Velocity at 1 AU: {U_solution[idx_1au]/1000:.1f} km/s")

    else:
        print("Solution failed:", sol.message)
except Exception as e:
    print(f"Error during integration: {e}")








