import numpy as np
import matplotlib.pyplot as plt

#Create figure with 1 row 3 collumns
fig1, ([ax1, ax2], [ax3, ax4]) = plt.subplots(2,2,figsize=(8,8))

#Defines data arrays for x and y (obtained during practical)
x_arr = [100,120,240,440,480,600,900,1200,1500,1800,2100,2400]
y_arr = [8.83,8.83,8.26,6.96,6.75,6.01,4.65,3.82,3.29,2.93,2.62,2.4]

#finds 3rd degree polynomial line that follows data closest
poly_coefficients = np.polyfit(x_arr, y_arr, 3)
smooth_x = np.linspace(100, 2400, 200) #Creates smooth x array with 200 even points between 100s and 2400s
smooth_y = np.polyval(poly_coefficients, smooth_x) #Subs smooth values of x into polynomial

#Finds rate based on polynomial values and obtained values seperately
smooth_rate = -np.gradient(smooth_y, smooth_x)
rate = -np.gradient(y_arr,x_arr)

#Finds rate slope and intercept when plotted against concentration
rate_slope, rate_intercept = np.polyfit(smooth_y, smooth_rate, 1)
rate_fit_line = rate_slope * smooth_y + rate_intercept #straight line equation for line of best fit of rate against concentration

#List of ln(conc)
ln_y = np.log(np.array(y_arr))
rate_slope2, rate_intercept2 = np.polyfit(x_arr, ln_y, 1)
rate_fit_line2 = rate_slope2 * np.array(x_arr) + rate_intercept2

#Plots smoothed curve and points used : concentration vs time
ax1.scatter(x_arr,y_arr, label = "Raw Data")
ax1.plot(smooth_x,smooth_y,label = "Curve fit")
ax1.set_title("Concentration against Time")
ax1.set_ylabel("Concentration / mols dm^-1")
ax1.set_xlabel("Time / s")
ax1.grid(True)
ax1.legend()

#Rate vs time
ax2.scatter(x_arr, rate, label= "Raw data")
ax2.plot(smooth_x, smooth_rate, label ="Curve fit")
ax2.set_title("Rate against Time")
ax2.set_ylabel("Rate / mols dm^-1 s^-1")
ax2.set_xlabel("Time / s")
ax2.grid(True)
ax2.legend()

#Rate vs concentration
ax3.scatter(y_arr,rate, label = "Raw data")
ax3.plot(smooth_y,rate_fit_line, label = f"y={round(rate_slope,5)}x + {round(rate_intercept,5)}")
ax3.set_title("Rate against Concentration")
ax3.set_ylabel("Rate / mols dm^-1 s^-1")
ax3.set_xlabel("Concentration / mols dm^-1")
ax3.grid(True)
ax3.legend()

#Ln conc against time
ax4.scatter(x_arr,ln_y, label = "Raw data")
ax4.plot(x_arr,rate_fit_line2,label = f"y={round(rate_slope2,5)}x + {round(rate_intercept2,5)}")
ax4.grid(True)
ax4.legend()

#Neat layout
plt.tight_layout()
plt.show()