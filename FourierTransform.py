import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
from CompositeWaveGenerator import FID_generator

# Number of samplepoints
N = 9000
# sample spacing
T = 1.0 / 8000 #Samples per second
x = np.linspace(0.0, N*T, N) #x axis for FID
y = 0 #empty y

#The above code was adapted from a stack overflow page, to help get me started

#spectrometer MHz
spectrometer_MHz = 300

y = FID_generator(x,y,N,spectrometer_MHz)

yf = scipy.fftpack.fft(y)
xf = np.linspace(y[0], 1.0/(2.0*T), N//2) #outputs FT in Hz
xf_ppm = xf*10 / spectrometer_MHz  #converts xf to ppm by dividing by Hz*e6

#Create a figure with 1 row and 2 columns of subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

#Plot the Pre-FT FID (time against signal intensity)
ax1.plot(x, y, color="blue")
ax1.set_title("Pre-FT: Time vs Signal")
ax1.set_xlabel("Time (seconds)")
ax1.set_ylabel("Amplitude")
ax1.set_xlim(0, max(x))  # Zoom into the first 0.75 seconds to see the shape
ax1.grid(True)

#Plot the Post-FT Frequency Domain Spectrum
ax2.plot(xf_ppm, 2.0/N * np.abs(yf[:N//2]), color="red")
ax2.set_xlim(0,13)
ax2.invert_xaxis()
ax2.yaxis.tick_right(),ax2.yaxis.set_label_position("right")
ax2.set_title("Post-FT: NMR Spectrum")
ax2.set_xlabel("Chemical shift (ppm)")
ax2.set_ylabel("Signal intensity")
ax2.grid(True)


#Display the two graphs side by side
plt.tight_layout()
plt.show()