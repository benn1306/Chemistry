import numpy as np
import matplotlib.pyplot as plt

def rydberg(n1,n2):
    E=1.096*10**7 * ((1/n1**2) - (1/n2**2))   #rydberg equation in code
    wavelength = 1/E #m
    wavelength = wavelength * 10**9
    return wavelength

print(rydberg(1,2),"m")
#gives wavelength 122 which is not visible

list_x = []

for n in range(2,11): # iterate through all jumps 1-10 starting at 1->2
    wave = rydberg(1,n)

    list_x.append(wave)
    
list_x = np.array(list_x)

plt.eventplot(list_x)
plt.xlabel('wavelength')
plt.show()


#as n increases the wavelength converges on the ionisation energy