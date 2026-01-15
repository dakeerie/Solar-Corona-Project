# Solar-Corona-Project

My work on solving the Parker solar wind equation using the RK4 method. The Parker equation is of the form

$\frac{1}{v} \frac{dv}{dr} = \frac{2c_s}{r}  \frac{1 - \frac{r_c}{r}}{v^2 - c_s^2}$ (1)

and describes the radial velocity profile, $v(r)$, of an isothermal solar wind. The physics of the system are characterised by the sound speed, $c_s = \sqrt{\frac{2 k_B T}{m_p}} = \sqrt{\frac{P}{\rho}}$, and the critical radius, $r_c = \frac{GM}{2c_s^2}$, at which $\frac{dv}{dr} = 0$ unless $v = c_s$. 

The Parker equation can be solved analytically to give 

$\frac{v^2}{c_s^2} - 2 \ln{\frac{v}{c_s}} = 4 \ln{\frac{r}{r_c}} + 4 \frac{r}{r_c} + const.$

The physically meaningful solution to the above equation is selected based on continuity of solutions and observations of the solar wind. This particular velocity solution is subsonic $(v<c_s)$ in the region $r<r_c$, crosses the critical radius, $r_c$, with $v = c_s$ and becomes supersonic $(v>c_s)$ in the region $r>r_c$. By setting $v = c_s$ at $r = r_c$, $const.$ is found to be equal to $3$. By using code from [Edward Villegas-Pulgarin](https://www.cosmoscalibur.com/en/blog/2024/espiral-de-parker-con-python/), I was able to create a solver which handles the transition through the critical point and produces results consistent with the analytical solution to the Parker equaiton.

The aim of the project was to create a numerical solver capable of handling the critical point with a view to extending the summer project into an MSci project where additional heating terms were to be added to model the non-isothermal nature of the solar wind. In the end I decided to pursue an MSci project in General Relativity instead of Solar Physics as that is where my true interest lies. I have put the experience I gained from this project to good use in numerically solving the Regge-Wheeler-Zerilli equation for my MSci project on black hole perturbations. 
