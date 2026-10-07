import numpy as np
import random
def FID_generator(x,y,N,spectrometer_MHz):
    #generating random FID with up to 8 peaks
    for i in range(0,random.randint(1,8)):
        #set peak intensity and frequency
        peak_intensity = random.randint(1,10)
        peak_freq = random.randint(10,350)

        #randomise splitting
        is_split = (random.randint(0,4) == 0)
        if is_split:
            splitting = random.randint(2,4)
            J = random.randint(5,15) #define J coupling if split
            
            if splitting == 2: #splitting for doublet 1:1
                y += (0.5 * peak_intensity) * np.sin((peak_freq + J/2.0) * 2.0*np.pi*x)
                y += (0.5 * peak_intensity) * np.sin((peak_freq - J/2.0) * 2.0*np.pi*x)
                print(f"Peak {i}: Doublet at {round(peak_freq*20/spectrometer_MHz,2)} ppm (J = {J} Hz)")
            
            if splitting == 3: #splitting for triplet 1:2:1
                y += (0.25 * peak_intensity) * np.sin((peak_freq + J) * 2.0*np.pi*x)
                y += (0.5 * peak_intensity) * np.sin((peak_freq) * 2.0*np.pi*x)
                y += (0.25 * peak_intensity) * np.sin((peak_freq - J) * 2.0*np.pi*x)
                print(f"Peak {i}: Triplet at {round(peak_freq*20/spectrometer_MHz,2)} ppm (J = {J} Hz)")
            
            if splitting == 4: #splitting for quartet 1:2:2:1
                y += (0.1667 * peak_intensity) * np.sin((peak_freq + 2*J) * 2.0*np.pi*x)
                y += (0.333 * peak_intensity) * np.sin((peak_freq + J) * 2.0*np.pi*x)
                y += (0.333 * peak_intensity) * np.sin((peak_freq - J) * 2.0*np.pi*x)
                y += (0.1667 * peak_intensity) * np.sin((peak_freq - 2*J) * 2.0*np.pi*x)
                print(f"Peak {i}: Quartet at {round(peak_freq*20/spectrometer_MHz,2)} ppm (J = {J} Hz)")
        
        else: #no splitting
            y += peak_intensity * np.sin(peak_freq * 2.0*np.pi*x)
            print(f"Peak {i}: Singlet at {round(peak_freq*20/spectrometer_MHz,2)} ppm")
    
    y += np.random.normal(0.0, 1, N)  #adding noise to FID
    y *= np.e**-(2.0*x**2)   #exponential decay to reduce noise (higher sensitivity, lower resolution), apodisation
    
    return y

#Function returns apodised randomly generated composite wave with up to 8 (or 32 if all quartets) random sin waves and random baseline noise