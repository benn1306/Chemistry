import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit # Added standard clean import for multi-step fitting

#function for R^2 value
def Coefficient_Of_Determination(y,line_eq):
    mean = np.average(y)
    SS_tot = np.sum((y-mean)**2) #Calculate total variation
    SS_res = np.sum((y-line_eq)**2)  #Calculate unexplained variation
    R_squared = round(1-(SS_res/SS_tot),3) #Calculate percent of explained variation (R^2 value)
    return R_squared
#function for common conc and rate graphs
def Common_Rate_Graphs(x_arr,y_arr):
    
    #finds 3rd degree polynomial line that follows data closest
    spline = UnivariateSpline(x_arr,y_arr)
    smooth_x = np.linspace(min(x_arr), max(x_arr), 200) #Creates smooth x array with 200 even points between 100s and 2400s
    smooth_y = spline(smooth_x) #Subs smooth values of x into polynomial

    #Finds rate based on polynomial values and obtained values seperately
    smooth_rate = -spline.derivative()(smooth_x)
    rate = -np.gradient(y_arr,x_arr)

    #Finds rate slope and intercept when plotted against concentration
    poly_coeffs = np.polyfit(smooth_y, smooth_rate, 2)
    rate_fit_line = np.polyval(poly_coeffs,smooth_y) #equation for line of best fit of rate against concentration
    a,b,c = round(poly_coeffs[0], 3), round(poly_coeffs[1], 3), round(poly_coeffs[2], 3)

    return [smooth_x,smooth_y,rate,smooth_rate,rate_fit_line,a,b,c]


# Integrated kinetic equation for consecutive 2-step first-order reaction
def consecutive_first_order(t, A0, k1, k2):#A0 = initial conc, k1 = A->B rate, k2 = B->C rate
    if np.isclose(k1, k2): k2 += 1e-5 #Prevent division by zero by adding small amount if k1 and k2 are too close
    return A0 * (k2 * np.exp(-k1 * t) - k1 * np.exp(-k2 * t)) / (k2 - k1) #returns full integrated rate equation


#Create figure with 1 row 3 collumns
fig1, ([ax1, ax2], [ax3, ax4]) = plt.subplots(2,2,figsize=(10,7))


# defines time values for x axis
x_arr = np.array([0, 1, 2, 3, 4, 5, 10])


#Allows testing of different data sets that should have 0,1,2 order respective to entered number
test_value = int(input("Enter number 0-3: "))
if test_value != 3:
    if test_value == 0:
        #Zero Order Target - example experimental data - uncomment line below to test
        y_true = np.clip(1.0 - 0.08 * x_arr, 0.01, None) 
    elif test_value == 1:
        #First Order Target - example experimental data - uncomment line below to test
        y_true = np.exp(-0.2 * x_arr) 
    elif test_value ==2:
        #Second Order Target - example experimental data - uncomment line below to test
        y_true = 1.0 / (1.0 + 0.3 * x_arr)
    #generate random noise
    experimental_scatter = np.random.normal(loc=0.0, scale=0.02, size=x_arr.shape)

    #Final y values with fake experimental error, clip ensures none below 0
    y_arr = list(np.clip(y_true + experimental_scatter, 0.04, None))

else: #custom values
    y_arr = np.array(input("y values array (seperated by spaces): ").split(),dtype=float)
    x_arr = np.array(input("x values array (seperated by spaces): ").split(),float)

rate_Graph = Common_Rate_Graphs(x_arr,y_arr)

#Calculates line of best fit and R^2 for zero order reaction
y = np.array(y_arr)
rate_slope0, rate_intercept0 = np.polyfit(x_arr, y, 1)
rate_fit_line0 = rate_slope0 * np.array(x_arr) + rate_intercept0
R_squared_zero_order = Coefficient_Of_Determination(y,rate_fit_line0)

#Calculates line of best fit and R^2 for first order reaction 
ln_y = np.log(np.array(y_arr))
rate_slope1, rate_intercept1 = np.polyfit(x_arr, ln_y, 1)
rate_fit_line1 = rate_slope1 * np.array(x_arr) + rate_intercept1
R_squared_first_order = Coefficient_Of_Determination(ln_y,rate_fit_line1)

#Calculates line of best fit and R^2 for second order reaction
over_y = 1 / np.array(y_arr)
rate_slope2, rate_intercept2 = np.polyfit(x_arr, over_y, 1)
rate_fit_line2 = rate_slope2 * np.array(x_arr) + rate_intercept2
R_squared_second_order = Coefficient_Of_Determination(over_y,rate_fit_line2)

#calculate if model is overfitting (prevent false positive for 2 step consecutive first order), uses akaike information criterion
def calc_aic(y_true,line_eq,num_params): #y_true = real data values, line_eq = predicted data values, num_params = number of parameters in line equation
    SS_res = np.sum((y_true - line_eq)**2) #calculate sum of square residuals
    if SS_res == 0:
        SS_res = 1e-10 #if fit is perfect avoid log 0 in next line
    N = len(y_true)
    return (N * np.log(SS_res / N) + 2 * num_params)
#when inputting data, ensure your y values and equation are both in native space (no log y or 1/y must be returned to just y)

#Which graph plots most accurate line of best fit
if R_squared_zero_order > R_squared_first_order and R_squared_zero_order > R_squared_second_order: #Check 0 order
    R_squared = R_squared_zero_order
    rate_fit_line,rate_slope,rate_intercept = rate_fit_line0,rate_slope0,rate_intercept0
    y_data = y
    order = ("0th order")
    integrated_y_label,integrated_x_label = ("concentration / mol dm^-1"),("time / s")
    best_aic = calc_aic(y,rate_fit_line0,2)
elif R_squared_first_order > R_squared_second_order:  #Check 1 order
    R_squared = R_squared_first_order
    rate_fit_line,rate_slope,rate_intercept = rate_fit_line1,rate_slope1,rate_intercept1
    y_data = ln_y
    order = ("1st order")
    integrated_y_label,integrated_x_label = ("ln(concentration)"),("time / s")
    best_aic = calc_aic(np.exp(ln_y),np.exp(rate_fit_line1),2)

else: #Check 2nd order
    R_squared = R_squared_second_order
    rate_fit_line,rate_slope,rate_intercept = rate_fit_line2,rate_slope2,rate_intercept2
    y_data = over_y
    order = ("2nd order")
    integrated_y_label,integrated_x_label = ("1/concentration / mol^-1 dm"),("time / s")
    best_aic = calc_aic(1/over_y,1/rate_fit_line2,2)
    print(2)
#Integrated rate equation for 2-step consecutive 1st order reaction
if test_value == 3: #only for inputted data
    try:
        #ensure first R^2 value generated higher than this to get save real baseline
        best_R2 = -1.0
        #defines array of best 3 parameters found
        best_guesses = [y[0] * 1.1, 0.001, 0.003]
        
        #Test a 9 combinations of starting speeds
        for test_k1 in [0.0005, 0.001, 0.002, 0.004]:
            for test_k2 in [0.0015, 0.003, 0.006, 0.012]:
                try:
                    popt_test, _ = curve_fit(consecutive_first_order, x_arr, y, p0=[y[0] * 1.1, test_k1, test_k2], bounds=(0, [y[0] * 2, 0.1, 0.1]), maxfev=1000)
                    fit_test = consecutive_first_order(x_arr, *popt_test)
                    r2_test = Coefficient_Of_Determination(y, fit_test)
                    if r2_test > best_R2: #if optimisation results from these parameters better, then use these instead
                        best_R2 = r2_test
                        best_guesses = popt_test
            
                except:
                    continue #if current guess caused error, continue

        #Once all optimisation runs complete, one final run with more steps to calculate better constants
        popt, _ = curve_fit(consecutive_first_order, x_arr, y, 
                                p0=best_guesses, 
                                bounds=(0, [y[0] * 2, 0.1, 0.1]), 
                                maxfev=10000)
        fit_values = consecutive_first_order(x_arr, *popt)
        R_squared_conc = Coefficient_Of_Determination(y, fit_values)
        aic_conc = calc_aic(y,fit_values,3)
        #Compare current lowest aic with consecutive aic to see if its a better fit (lower is better)
        if aic_conc < best_aic:
            R_squared = R_squared_conc
            order = "Consecutive 2-Step 1st Order"
            y_data = y
            integrated_y_label = "concentration / mol dm^-3"
            rate_fit_line = fit_values
            rate_slope, rate_intercept = popt[1], popt[2]
            
    except Exception as e:
        print(f"Optimization failed to converge: {e}") #crash protection



#Plots smoothed curve and points used : concentration vs time
ax1.scatter(x_arr,y_arr, label = "Raw Data")
ax1.plot(rate_Graph[0],rate_Graph[1],label = "Curve fit")
ax1.set_title("Concentration against Time")
ax1.set_ylabel("Concentration / mols dm^-1")
ax1.set_xlabel("Time / s")
ax1.grid(True)
ax1.legend()

#Rate vs time
ax2.scatter(x_arr, rate_Graph[2], label= "Raw data")
ax2.plot(rate_Graph[0], rate_Graph[3], label ="Curve fit")
ax2.set_title("Rate against Time")
ax2.set_ylabel("Rate / mols dm^-1 s^-1")
ax2.set_xlabel("Time / s")
ax2.grid(True)
ax2.legend()

#Rate vs concentration
ax3.scatter(y_arr,rate_Graph[2], label = "Raw data")
ax3.plot(rate_Graph[1],rate_Graph[4], label = f"y={round(rate_Graph[5],5)}x^2 + {round(rate_Graph[6],5)}x + {round(rate_Graph[7],5)}")
ax3.set_title("Rate against Concentration")
ax3.set_ylabel("Rate / mols dm^-1 s^-1")
ax3.set_xlabel("Concentration / mols dm^-1")
ax3.invert_xaxis()
ax3.grid(True)
ax3.legend()


#Integrated rate equation
ax4.scatter(x_arr,y_data, label = "Raw data")
ax4.plot(x_arr,rate_fit_line,label = f"y={round(rate_slope,5)}x + {round(rate_intercept,5)}\nR^2 = {R_squared}")
ax4.set_title(f"Integrated rate equation {order} reaction")
ax4.set_ylabel(f"{integrated_y_label}")
ax4.set_xlabel(f"{integrated_x_label}")
ax4.grid(True)
ax4.legend()

#Neat layout
plt.tight_layout()
plt.show()
