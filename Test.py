import astropy.constants
import scipy
import astropy

print(scipy.constants.c)

print(astropy.constants.M_sun)

mass = 10e5
print(2*scipy.constants.G*mass*1.988e30/scipy.constants.c**2)