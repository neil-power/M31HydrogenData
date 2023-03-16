#IMPORTS AND DATA ---------------------------------------------------------------------------------------------------------------

import os
import math
import numpy as np #Import numpy for maths calculations
import matplotlib.pyplot as plt #Import pyplot for plotting the graph
from scipy.integrate import simpson

MAX_DEGREE = 5
PEAK_WIDTH = 40
S8_FLUX = 850 #K km/s
PD = "b-" #Format of plotted data - see pyplot docs for parameters
FD = "r-" #Format of fitted line - see pyplot docs for parameters

crop_data = np.genfromtxt("CropPositions.txt",dtype=str,delimiter=",")
crop_params = {}

int_lim_data = np.genfromtxt("IntegralBounds.txt",dtype=str,delimiter=",")
int_params = {}

gaussian_data = np.genfromtxt("GaussianPositions.txt",dtype=str,delimiter=",")
gaussian_params = {}

for i in range(0,np.shape(crop_data)[0]):
    crop_line = crop_data[i]
    crop_params[crop_line[0]] = crop_line[1:]
    
for i in range(0,np.shape(int_lim_data)[0]):
    int_line = int_lim_data[i]
    int_params[int_line[0]] = int_line[1:]
    
    gauss_line = gaussian_data[i]
    gaussian_params[gauss_line[0]] = gauss_line[1:]

def remove_peaks(x,y,title):
    crops = crop_params[title]
    peaks = crops[0].split(" ")
    widths = crops[1].split(" ")
    boolean_to_remove = np.full(np.size(y),True)

    for i in range(0,len(peaks)):
        max_position = int(peaks[i])
        width_to_remove = int(widths[i])*2
        boolean_to_remove[max_position-width_to_remove:max_position+width_to_remove] = False
    return x[boolean_to_remove],y[boolean_to_remove]

def root_mean_square(x,y,fit):
       mean_sq = np.sqrt(np.sum(np.abs(y-np.polyval(fit, x))**2)/np.size(y))
       return mean_sq

def calibrate_elevation(flux,elevation):
    calibrated_flux = flux*np.exp(0.0076*(1/(np.sin(math.radians(elevation)))-1))
    return calibrated_flux

def calibrate_flux(path):
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

    cropped_velocity,cropped_flux = remove_peaks(velocity,elevation_calibrated_flux,title)

    fit = fit_equation(cropped_velocity,cropped_flux)
    flux_without_baseline = elevation_calibrated_flux - np.polyval(fit, velocity)

    area = simpson(flux_without_baseline, velocity)
    return S8_FLUX/area

def fit_equation(x,y): #Fits a poly equation to data
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


def plot_velocity_baseline(velocity,flux,title):

    #VISUALS ---------------------------------------------------------------------------------------------------------------

    plt.rcParams["figure.figsize"] = (7,5) #Set
    plt.xlabel('Velocity [kms^-1]')
    plt.ylabel('Uncalibrated Flux')
    plt.title(title+ ' Velocity-Uncorrected Flux Plot')

    #PLOTTING DATA ------------------------------------------------------------

    cropped_velocity,cropped_flux = remove_peaks(velocity,flux,title)

    crops = crop_params[title]
    peaks = np.array(crops[0].split(" ")).astype(int)

    plt.plot(velocity[peaks],flux[peaks],"r.",zorder=0)

    fit = fit_equation(cropped_velocity,cropped_flux)

    plt.plot(cropped_velocity,cropped_flux,"g--",zorder=10)

    plt.plot(velocity,np.polyval(fit, velocity), FD,zorder=5)

    plt.plot(velocity,flux,PD,zorder=0)

    plt.legend(["Peaks","Peak Removed Data", "Fit","Data"]) #Add a legend to the graph
    #print(np.poly1d(fit.round(decimals=5)))
    #plt.savefig("Graphs\\"+title +"DegreesFrequency.png", dpi=600) #Saves graph
    plt.show() #Displays graph
    #plt.clf()


def plot_velocity_without_baseline(velocity,flux,title):

    #VISUALS ---------------------------------------------------------------------------------------------------------------

    plt.rcParams["figure.figsize"] = (7,5) #Set the size of the final graph to 15x5 inches
    plt.xlabel('Velocity [kms^-1]')
    plt.ylabel('Uncalibrated Flux')
    plt.title(title+ ' Velocity-Uncorrected Flux Plot (Background Corrected)')

    #PLOTTING DATA ---------------------------------------------------------------------------------------------------------------
    #plt.ylim([-1,+1])
    plt.plot(velocity, flux,PD,zorder=0)
    plt.legend(["Data"])

    #plt.savefig("Graphs\\"+title +"DegreesVelocity.png", dpi=600) #Saves graph
    plt.show() #Displays graph
    plt.clf()

def plot_velocity_without_baseline_corrected(velocity,flux,title):

    #VISUALS ---------------------------------------------------------------------------------------------------------------

    plt.rcParams["figure.figsize"] = (10,8) #Set the size of the final graph to 15x5 inches
    plt.xlabel('Velocity [kms^-1]')
    plt.ylabel('Flux [K]')
    plt.title(title+ ' Velocity-Flux Plot (Corrected)')

    #PLOTTING DATA ---------------------------------------------------------------------------------------------------------------
    #plt.ylim([-1,+1])
    plt.plot(velocity, flux,PD,zorder=0)
    plt.legend(["Data"])

    #plt.savefig("Graphs\\"+title +"DegreesVelocity.png", dpi=600) #Saves graph
    plt.show() #Displays graph
    #plt.clf()
    
def gaussian(x,a,b,c):
    return a*np.exp( -( (x-b)**2 )/(2*c**2))
    
def plot_velocity_for_integration(velocity,flux,title):
    if int_params[title][2] == "1":
        print("Separate Peak")
        min_bound = int(int_params[title][0])
        max_bound = int(int_params[title][1])
        plt.plot([velocity[min_bound],velocity[min_bound]],[0,max(flux)],"r--")
        plt.plot([velocity[max_bound],velocity[max_bound]],[0,max(flux)],"r--")
    
    #VISUALS ---------------------------------------------------------------------------------------------------------------

    plt.rcParams["figure.figsize"] = (7,5) #Set the size of the final graph to 15x5 inches
    plt.xlabel('Velocity [kms^-1]')
    plt.ylabel('Flux [K]')
    plt.title(title+ ' Velocity-Flux Plot (Corrected)')
    plt.xlim([-150,100])
    #PLOTTING DATA ---------------------------------------------------------------------------------------------------------------
    plt.plot(velocity, flux,PD,zorder=0)
    if True: #int_params[title][2] == "0":
        crops = gaussian_params[title]
        peaks = np.array(crops[0].split(" ")).astype(float)
        widths = np.array(crops[1].split(" ")).astype(float)
        amps = np.array(crops[2].split(" ")).astype(float)
    
        peaks_to_subtract = np.full(np.size(velocity),0)
    
        for i in range(0,len(peaks)):
            a = amps[i] 
            b = peaks[i] 
            c = widths[i] 
            plt.plot(velocity,gaussian(velocity,a,b,c))
            peaks_to_subtract = np.maximum(gaussian(velocity,a,b,c),peaks_to_subtract)
    
        plt.plot(velocity, flux-peaks_to_subtract,"r--",zorder=5)
    print(simpson(flux-peaks_to_subtract, velocity))
    plt.show() #Displays graph
    plt.clf()

def plot_all_graphs():
    for entry in os.scandir("M31Backup\\"):
        title = entry.path[10:-4]
        if title.startswith("M"):
            analyse(entry.path)

def analyse(path,rtn=False):
    title = path[10:-4]
    metadata = np.genfromtxt(path,delimiter='=',skip_header=0, skip_footer=514,dtype=str)
    velocity_addition = float(metadata[-1][-1])
    bin_width = float(metadata[-5][-1])
    elevation = float(metadata[-11][-1])
    data = np.genfromtxt(path,dtype='float',delimiter='/',skip_header=27, skip_footer=1)

    scaled_velocity = np.array(data[:,1])
    uncalibrated_flux =  np.array(data[:,0])

    velocity = scaled_velocity*bin_width + velocity_addition
    
    #ORIGINAL
    #plot_velocity_baseline(velocity,uncalibrated_flux,title)
    
    #REMOVE BASELINE
    cropped_velocity,cropped_flux = remove_peaks(velocity,uncalibrated_flux,title)
    fit = fit_equation(cropped_velocity,cropped_flux)
    fit = np.polyfit(cropped_velocity,cropped_flux,deg=4)
    
    flux_without_baseline = uncalibrated_flux - np.polyval(fit, velocity)
    
    #plot_velocity_without_baseline(velocity,flux_without_baseline,title)
    
    #CALIBRATE FOR ELEVATION
    elevation_calibrated_flux = calibrate_elevation(flux_without_baseline, elevation)
    #plot_velocity_without_baseline(velocity,flux_without_baseline,title)

    #plot_velocity_without_baseline(velocity,elevation_calibrated_flux,title)
    
    #SCALE BRIGTNESS
    calibrated_flux = elevation_calibrated_flux*FLUX_CALIBRATION

    #plot_velocity_without_baseline_corrected(velocity,calibrated_flux,title)

    if rtn:
        return velocity,calibrated_flux
    
def save_peaks_by_index(velocity,flux,title):
    crops = crop_params[title]
    peaks = crops[0].split(" ")
    widths = crops[1].split(" ")
   
    peak_string = ""
    width_string = ""
    amp_string = ""
    
    for i in range(0,len(peaks)):
        max_position = int(peaks[i])
        width = int(widths[i])
        a = flux[max_position] #Height of curve's peak in K kms^-1
        b = velocity[int(peaks[i])] #Position of centre of peak in km/s
        c = width #np.abs(velocity[max_position-width] - velocity[max_position+width]) #Width peak in km/s

        peak_string += str(round(b,1)) + " "
        width_string += str(round(c,1)) + " "
        amp_string += str(round(a,1)) + " "

    with open("GaussianPositions_new.txt","a") as f:
        f.write(title + ","+peak_string.strip()+","+width_string.strip()+","+amp_string.strip()+"\n")
        
    
def match_coords():
    file_coordinates = {}
    for entry in os.scandir("M31Backup\\"):
        title = entry.path[10:-4]
        if title.startswith("M"):
            title = entry.path[10:-4]
            metadata = np.genfromtxt(entry.path,delimiter='=',skip_header=0, skip_footer=514,dtype=str)
            ra = float(metadata[5][-1])
            dec = float(metadata[6][-1])
            file_coordinates[title] = [ra,dec]
    return file_coordinates
   
                
def plot_coordinates():
    ra_coords = np.empty(0)
    dec_coords = np.empty(0)
    titles = []
    for entry in os.scandir("M31Backup\\"):
        title = entry.path[10:-4]
        if title.startswith("M"):
            title = entry.path[10:-4]
            metadata = np.genfromtxt(entry.path,delimiter='=',skip_header=0, skip_footer=514,dtype=str)
            ra = float(metadata[5][-1])
            dec = float(metadata[6][-1])
            titles.append(title[4:])
            ra_coords = np.hstack((ra_coords, ra))
            dec_coords = np.hstack((dec_coords, dec))
            
    plt.xlabel('Right Ascension [degrees]')
    plt.ylabel('Declination [degrees]')
    plt.title("Coordinates")
    plt.plot(ra_coords,dec_coords,"b.")
    plt.gca().invert_xaxis()
    for i,text in enumerate(titles):
        plt.annotate(text,(ra_coords[i],dec_coords[i]))
    plt.show()

def integrate_all_graphs():
    for entry in os.scandir("M31Backup\\"):
        title = entry.path[10:-4]
        if title.startswith("M"):
            velocity,flux = analyse(entry.path,True)
            plot_velocity_for_integration(velocity,flux,title)
            #save_peaks_by_index(velocity,flux,title)
     
#plot_coordinates()
#get elevations

FLUX_CALIBRATION = calibrate_flux("M31Backup\\S8A.TXT")
FILE_COORDS = match_coords()
#print(FLUX_CALIBRATION)

individual_path = "M31Backup\\M31P16.TXT"
velocity,flux = analyse(individual_path,True)
plot_velocity_for_integration(velocity,flux,individual_path[10:-4])
#plot_all_graphs()
#integrate_all_graphs()
