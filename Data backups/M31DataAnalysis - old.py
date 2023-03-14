#IMPORTS AND DATA ---------------------------------------------------------------------------------------------------------------

import numpy as np #Import numpy for maths calculations
import matplotlib.pyplot as plt #Import pyplot for plotting the graph
#from scipy import optimize
import os
import math
from scipy.integrate import simpson
from scipy import signal
from numpy import trapz

MAX_DEGREE = 5
PEAK_WIDTH = 40
S8_FLUX = 850*1e3 #K km/s
PD = "b-" #Format of plotted data - see pyplot docs for parameters
FD = "r-" #Format of fitted line - see pyplot docs for parameters

def root_mean_square(x,y,fit):
    mean_sq = np.sqrt(np.sum(np.abs(y-np.polyval(fit, x))**2)/np.size(y))
    return mean_sq

def remove_largest_peaks(x,y):
    found_peaks = signal.find_peaks(y,height = np.max(y), distance = 1,width=2)
    peak_widths = signal.peak_widths(y,found_peaks[0],rel_height=0.6)[0]
    #print("Peak: ",found_peaks[0])
    #print("Width: ", peak_widths)
    #print("Value: ", x[found_peaks[0]])
  
    boolean_to_remove = np.full(np.size(y),True)
    
    for i in range(0,len(found_peaks[0])):
        # print("removed")
        max_position = found_peaks[0][i]
        width_to_remove = int(peak_widths[i]*2)
        #     #print("cropped")
        boolean_to_remove[max_position-width_to_remove:max_position+width_to_remove] = False
        #x = np.hstack((x[:max_position-width_to_remove],x[max_position+width_to_remove:]))
        #y = np.hstack((y[:max_position-width_to_remove],y[max_position+width_to_remove:]))
  
    return x[boolean_to_remove],y[boolean_to_remove]

def remove_smallest_peaks(x,y):
    found_peaks = signal.find_peaks(y,height = np.max(y), distance = 1,width=2)
    peak_widths = signal.peak_widths(y,found_peaks[0],rel_height=0.2)[0]
    
  
    boolean_to_remove = np.full(np.size(y),True)
    
    for i in range(0,len(found_peaks[0])):
   
        max_position = found_peaks[0][i]
        width_to_remove = int(peak_widths[i])
        boolean_to_remove[max_position-width_to_remove:max_position+width_to_remove] = False
  
    return x[boolean_to_remove],y[boolean_to_remove]

def remove_negative_peaks(x,y):
    found_peaks = signal.find_peaks(y,height = np.min(y), distance = 1,width=2)
    peak_widths = signal.peak_widths(y,found_peaks[0],rel_height=0.2)[0]
    
  
    boolean_to_remove = np.full(np.size(y),True)
    
    for i in range(0,len(found_peaks[0])):
   
        max_position = found_peaks[0][i]
        width_to_remove = int(peak_widths[i])
        boolean_to_remove[max_position-width_to_remove:max_position+width_to_remove] = False
  
    return x[boolean_to_remove],y[boolean_to_remove]

def remove_area_around_peak(x,y,width_to_remove=PEAK_WIDTH):

    #min_height = np.max(y)/4
 
    x, y=remove_largest_peaks(x,y)
    #print(max(y))
    #print(np.std(y))
    while np.max(y) > 1.2*np.average(np.abs(y)):
        print("more peaks")
        x,y= remove_smallest_peaks(x,y)
    
    #if min(y) < -7:
        #Remove negative

    """   
    #print(np.max(np.abs(y)))
    print("STD 1: ", np.std(y))
    #Remove first peak
    max_position = np.argmax(np.abs(y))
    #     #print("cropped")
    x = np.hstack((x[:max_position-width_to_remove],x[max_position+width_to_remove:]))
    y = np.hstack((y[:max_position-width_to_remove],y[max_position+width_to_remove:]))
    
    if np.argmax(np.abs(y)) > 2*np.average(np.abs(y)):
        width_to_remove = 20
        max_position = np.argmax(np.abs(y))
        print("cropped x2")
        x = np.hstack((x[:max_position-width_to_remove],x[max_position+width_to_remove:]))
        y = np.hstack((y[:max_position-width_to_remove],y[max_position+width_to_remove:]))
    if np.argmax(np.abs(y)) > 2*np.average(np.abs(y)):
        print("Extra peak found?")
    #     width_to_remove = 20
    #     max_position = np.argmax(np.abs(y))
    #     print("cropped x3")
    #     x = np.hstack((x[:max_position-width_to_remove],x[max_position+width_to_remove:]))
    #     y = np.hstack((y[:max_position-width_to_remove],y[max_position+width_to_remove:]))
    x = x[y<1]
    y = y[y<1]
    """

    return x,y
    

def calibrate_elevation(flux,elevation):
    calibrated_flux = flux*np.exp(0.0076*(1/(np.cos(math.radians(elevation)))-1))
    return calibrated_flux

def calibrate_flux(path):
    metadata = np.genfromtxt(path,delimiter='=',skip_header=0, skip_footer=514,dtype=str)
    velocity_addition = float(metadata[-1][-1])
    bin_width = float(metadata[-5][-1])
    elevation = float(metadata[-11][-1]) 
    data = np.genfromtxt(path,dtype='float',delimiter='/',skip_header=27, skip_footer=1)
    
    scaled_velocity = np.array(data[:,1])
    uncalibrated_flux =  np.array(data[:,0])
    velocity = scaled_velocity*bin_width + velocity_addition
    elevation_calibrated_flux = calibrate_elevation(uncalibrated_flux, elevation)
    
    
    cropped_velocity,cropped_flux = remove_area_around_peak(velocity,elevation_calibrated_flux)
    
    fit = fit_equation(cropped_velocity,cropped_flux)
    flux_without_baseline = elevation_calibrated_flux - np.polyval(fit, velocity)
    
    interval = np.abs(velocity[0]-velocity[1])
    area = trapz(flux_without_baseline, velocity)
    print("area =", area)

    # Compute the area using the composite Simpson's rule.
    print(len(flux_without_baseline))
    print(len(velocity))
    area = simpson(flux_without_baseline, velocity)
    print("area =", area)
    return S8_FLUX/area
    
def fit_equation(x,y): #Fits a poly equation to data
    #optimize.fmin(reduced_chi_square)
    best_fit = 1
    best_res = 10000000
    for degree in range(1,MAX_DEGREE):
        curve = np.polyfit(x,y,degree,cov=False)
        red_chi = root_mean_square(x,y,curve)
        if red_chi < best_res:
            best_res = red_chi
            best_fit = degree
    print(best_fit)
    #print(best_res)
    return np.polyfit(x,y,best_fit,cov=False)


def plot_velocity_baseline(velocity,flux,title):

    #VISUALS ---------------------------------------------------------------------------------------------------------------

    plt.rcParams["figure.figsize"] = (7,5) #Set
    plt.xlabel('Velocity [ms^-1]')
    plt.ylabel('Uncalibrated Flux')
    plt.title(title+ ' Velocity-Uncorrected Flux Plot')

    #PLOTTING DATA ------------------------------------------------------------
    #print(title)
    #print(np.std(flux))
    #plt.xlim([-50,50])
    cropped_velocity,cropped_flux = remove_area_around_peak(velocity,flux)
    found_peaks = signal.find_peaks(flux,height = np.max(flux)/2, distance = 1,width=2)[0]
    plt.plot(velocity[found_peaks],flux[found_peaks],"r.")
    fit = fit_equation(cropped_velocity,cropped_flux)
    #print(np.std(cropped_flux))
    #print()
    plt.plot(cropped_velocity,cropped_flux,"g--",zorder=10)
    
    found_peaks = signal.find_peaks(cropped_flux,height = np.max(cropped_flux), distance = 1,width=2)[0]
    plt.plot(cropped_velocity[found_peaks],cropped_flux[found_peaks],"y.")
    
    plt.plot(velocity,np.polyval(fit, velocity), FD,zorder=5)
    
    plt.plot(velocity,flux,PD,zorder=0)
    
    plt.legend(["Large Peaks","Peak Removed Data","Small Peaks", "Fit","Data"]) #Add a legend to the graph
    #print(np.poly1d(fit.round(decimals=5)))
    #plt.savefig("Graphs\\"+title +"DegreesFrequency.png", dpi=600) #Saves graph
    plt.show() #Displays graph
    plt.clf()


def plot_velocity_without_baseline(velocity,flux,title):

    #VISUALS ---------------------------------------------------------------------------------------------------------------

    plt.rcParams["figure.figsize"] = (7,5) #Set the size of the final graph to 15x5 inches
    plt.xlabel('Velocity [ms^-1]')
    plt.ylabel('Uncalibrated Flux')
    plt.title(title+ ' Velocity-Uncorrected Flux Plot (Background Corrected)')

    #PLOTTING DATA ---------------------------------------------------------------------------------------------------------------
    #plt.ylim([-1,+1])
    plt.plot(velocity, flux,PD,zorder=0)
    plt.legend(["Velocity Scan"])

    #plt.savefig("Graphs\\"+title +"DegreesVelocity.png", dpi=600) #Saves graph
    plt.show() #Displays graph
    plt.clf()

def plot_all_graphs():
    for entry in os.scandir("M31Backup\\"):
        if entry.path.startswith("M"):
            analyse(entry.path)

def analyse(path):
    title = path[10:-4]
    metadata = np.genfromtxt(path,delimiter='=',skip_header=0, skip_footer=514,dtype=str)
    velocity_addition = float(metadata[-1][-1])
    bin_width = float(metadata[-5][-1])
    elevation = float(metadata[-11][-1]) 
    data = np.genfromtxt(path,dtype='float',delimiter='/',skip_header=27, skip_footer=1)
    
    scaled_velocity = np.array(data[:,1])
    uncalibrated_flux =  np.array(data[:,0])
    
    velocity = scaled_velocity*bin_width + velocity_addition
    elevation_calibrated_flux = calibrate_elevation(uncalibrated_flux, elevation)
    
    #plot_velocity_baseline(velocity,uncalibrated_flux,title)
    
    plot_velocity_baseline(velocity,elevation_calibrated_flux,title)
    
    #plot_velocity_without_baseline(velocity,elevation_calibrated_flux,title)
    
    cropped_velocity,cropped_flux = remove_area_around_peak(velocity,elevation_calibrated_flux)
    fit = fit_equation(cropped_velocity,cropped_flux)
    
    flux_without_baseline = elevation_calibrated_flux - np.polyval(fit, velocity)
    calibrated_flux = flux_without_baseline* FLUX_CALIBRATION
    
    #plot_velocity_without_baseline(velocity,calibrated_flux,title)

FLUX_CALIBRATION = calibrate_flux("M31Backup\\S8A.TXT")
print(FLUX_CALIBRATION)
#print(calibrate_flux("M31Backup\\S8B.TXT"))
#analyse("M31Backup\\M31P85.TXT")
#analyse("M31Backup\\S8A.TXT")
analyse("M31Backup\\M31P110.TXT")
#analyse("M31Backup\\M31P20.TXT")
#plot_all_graphs()



