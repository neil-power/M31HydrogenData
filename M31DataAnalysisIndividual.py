#IMPORTS AND DATA ---------------------------------------------------------------------------------------------------------------

import numpy as np #Import numpy for maths calculations
import matplotlib.pyplot as plt #Import pyplot for plotting the graph
#from scipy import optimize
import os
from scipy import signal

MAX_DEGREE = 5
PEAK_WIDTH = 40
S8_FLUX = 850*1e3 #K km/s
PD = "b-" #Format of plotted data - see pyplot docs for parameters
FD = "r-" #Format of fitted line - see pyplot docs for parameters

def root_mean_square(x,y,fit):
    mean_sq = np.sqrt(np.sum(np.abs(y-np.polyval(fit, x))**2)/np.size(y))
    return mean_sq

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
    #print(best_fit)
    #print(best_res)
    return np.polyfit(x,y,best_fit,cov=False)


def remove_peaks(x,y,found_peaks,peak_widths):


    boolean_to_remove = np.full(np.size(y),True)

    for i in range(0,len(found_peaks)):
        # print("removed")
        max_position = found_peaks[i]
        width_to_remove = int(peak_widths[i]*2)
        #     #print("cropped")
        boolean_to_remove[max_position-width_to_remove:max_position+width_to_remove] = False


    return x[boolean_to_remove], y[boolean_to_remove]

def plot_velocity_baseline(velocity,flux,title):

    #VISUALS ---------------------------------------------------------------------------------------------------------------

    plt.rcParams["figure.figsize"] = (7,5) #Set
    plt.xlabel('Velocity [ms^-1]')
    plt.ylabel('Uncalibrated Flux')
    plt.title(title+ ' Velocity-Uncorrected Flux Plot')

    #PLOTTING DATA ------------------------------------------------------------
    #print(title)

    found_peaks = signal.find_peaks(flux,height = np.max(flux)/8, distance = 1,width=2)[0]
    peak_widths = signal.peak_widths(flux,found_peaks,rel_height=0.6)[0]
    
    #PEAKS BY INDEX

    peak_string = ""
    width_string = ""
    for i in range(0,len(found_peaks)):
        print("Peak ",i,": ",found_peaks[i])
        print("Width: ",int(peak_widths[i]))
        peak_string += str(int(found_peaks[i])) + " "
        width_string += str(int(peak_widths[i])) + " "

    with open("CropPositions_new.txt","a") as f:
        f.write(title + ","+peak_string.strip()+","+width_string.strip()+"\n")
        
    #PEAKS BY VALUE for gaussian

    # peak_string = ""
    # width_string = ""
    # amp_string = ""
    # for i in range(0,len(found_peaks)):
    #     print("Peak ",i,": ",found_peaks[i])
    #     print("Width: ",int(peak_widths[i]))
    #     peak_string += str(int(found_peaks[i])) + " "
    #     width_string += str(int(peak_widths[i])) + " "

    # with open("GaussianPositions_new.txt","a") as f:
    #     f.write(title + ","+peak_string.strip()+","+width_string.strip()+","+amp_string.strip()+"\n")

    """
    plt.plot(velocity[found_peaks],flux[found_peaks],"r.")

    cropped_velocity,cropped_flux = remove_peaks(velocity,flux,found_peaks,peak_widths)
    fit = fit_equation(cropped_velocity,cropped_flux)

    plt.plot(cropped_velocity,cropped_flux,"g--",zorder=10)


    plt.plot(velocity,np.polyval(fit, velocity), FD,zorder=5)

    plt.plot(velocity,flux,PD,zorder=0)

    plt.legend(["Large Peaks","Peak Removed Data","Small Peaks", "Fit","Data"]) #Add a legend to the graph
    #print(np.poly1d(fit.round(decimals=5)))
    #plt.savefig("Graphs\\"+title +"DegreesFrequency.png", dpi=600) #Saves graph
    plt.show() #Displays graph
    plt.clf()

    """
def plot_all_graphs():
    for entry in os.scandir("M31Backup\\"):
        if entry.path.startswith("M"):
            analyse(entry.path)
            # check = input()
            # while input() != "y":
            #     check = input()

def save_peaks_by_index(velocity,flux,title):
    crops = crop_params[title]
    peaks = crops[0].split(" ")
    widths = crops[1].split(" ")
   
    peak_string = ""
    width_string = ""
    amp_string = ""
    
    for i in range(0,len(peaks)):
        if velocity[int(peaks[i])] > 0:
                max_position = int(peaks[i])
                width = int(widths[i])
                a = flux[max_position] #Height of curve's peak in K kms^-1
                b = velocity[int(peaks[i])] #Position of centre of peak in km/s
                c = width #np.abs(velocity[max_position-width] - velocity[max_position+width]) #Width peak in km/s

                peak_string += str(b) + " "
                width_string += str(c) + " "
                amp_string += str(a) + " "

    with open("GaussianPositions_new.txt","a") as f:
        f.write(title + ","+peak_string.strip()+","+width_string.strip()+","+amp_string.strip()+"\n")


def analyse(path):
    title = path[10:-4]
    metadata = np.genfromtxt(path,delimiter='=',skip_header=0, skip_footer=514,dtype=str)
    velocity_addition = float(metadata[-1][-1])
    bin_width = float(metadata[-5][-1])
    data = np.genfromtxt(path,dtype='float',delimiter='/',skip_header=27, skip_footer=1)

    scaled_velocity = np.array(data[:,1])
    uncalibrated_flux =  np.array(data[:,0])

    velocity = scaled_velocity*bin_width + velocity_addition


    plot_velocity_baseline(velocity,uncalibrated_flux,title)

open("CropPositions_new.txt", 'w').close() #clear file
open("GaussianPositions_new.txt", 'w').close() #clear file
#print(calibrate_flux("M31Backup\\S8B.TXT"))
#analyse("M31Backup\\M31P85.TXT")
#analyse("M31Backup\\S8A.TXT")
#analyse("M31Backup\\M31P37.TXT")
#analyse("M31Backup\\M31P110.TXT")
plot_all_graphs()



