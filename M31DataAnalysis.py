#IMPORTS -------- ----------------------------------------------------------

import os
import math
import numpy as np #Import numpy for maths calculations
import matplotlib.pyplot as plt #Import pyplot for plotting the graph
from scipy.integrate import simpson

MAX_DEGREE = 5
PEAK_WIDTH = 40
S8_FLUX = 850 #K km/s
PD = "b-" 
FD = "r-"

# READ IN DATA -------------------------------------------------------------

crop_params = {}
int_params = {}
gaussian_params = {}

def read_in_data():
    crop_data = np.genfromtxt("CropPositions.txt",dtype=str,delimiter=",")
    int_lim_data = np.genfromtxt("IntegralBounds.txt",dtype=str,delimiter=",")
    gaussian_data = np.genfromtxt("GaussianPositions.txt",dtype=str,delimiter=",")

    for i in range(0,np.shape(crop_data)[0]):
        crop_line = crop_data[i]
        crop_params[crop_line[0]] = crop_line[1:]

    for i in range(0,np.shape(int_lim_data)[0]):
        int_line = int_lim_data[i]
        int_params[int_line[0]] = int_line[1:]

        gauss_line = gaussian_data[i]
        gaussian_params[gauss_line[0]] = gauss_line[1:]

read_in_data()

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

# DEFINE MATHS FUNCTIONS --------------------------------------------------

def root_mean_square(x,y,fit):
    mean_sq = np.sqrt(np.sum(np.abs(y-np.polyval(fit, x))**2)/np.size(y))
    return mean_sq

def fit_equation(x,y): #Fits a poly equation to data
    best_fit = 1
    best_res = 10000000
    for degree in range(1,MAX_DEGREE):
        curve = np.polyfit(x,y,degree,cov=False)
        red_chi = root_mean_square(x,y,curve)
        if red_chi < best_res:
            best_res = red_chi
            best_fit = degree
    return np.polyfit(x,y,best_fit,cov=False)

def gaussian(x,a,b,c):
    return a*np.exp( -( (x-b)**2 )/(2*c**2))

def calibrate_elevation(flux,elevation):
    calibrated_flux = flux*np.exp(0.0076*(1/(np.sin(math.radians(elevation)))-1))
    return calibrated_flux

# CONSTANTS CALCULATIONS --------------------------------------------------

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

def get_average_milkyway(plot=False):
    n = 0
    total_array = np.zeros(shape=(511,))
    for entry in os.scandir("M31Backup\\"):
        title = entry.path[10:-4]
        if title.startswith("M"):
            if int(title[4:]) < 80:
                velocity,flux = calibrate(entry.path,True)
                total_array[370:420] += flux[370:420]
                n += 1
    total_array /= n
    if plot:
        plot_velocity_without_baseline_corrected(velocity, total_array,"Average Milky Way")
    return total_array

#REMOVING PEAKS ------------------------------------------

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

    with open("GaussianPositions_new.txt","a",encoding="utf-8") as f:
        f.write(title + ","+peak_string.strip()+","+width_string.strip()+","+amp_string.strip()+"\n")

#REMOVING LOCAL HYDROGEN ---------------------------------------------------

def remove_local_hydrogen(velocity,flux,title):
    peaks_to_subtract = np.full(np.size(velocity),0)
    crops = gaussian_params[title]
    peaks = np.array(crops[0].split(" ")).astype(float)
    widths = np.array(crops[1].split(" ")).astype(float)
    amps = np.array(crops[2].split(" ")).astype(float)

    for i in range(0,len(peaks)):
        a = amps[i]
        b = peaks[i]
        c = widths[i]
        plt.plot(velocity,gaussian(velocity,a,b,c))
        peaks_to_subtract = np.maximum(gaussian(velocity,a,b,c),peaks_to_subtract)

    return velocity, flux-peaks_to_subtract

def remove_local_hydrogen_by_average(velocity,flux):
    return velocity, flux- AVG_MILKY_WAY
    #*((AVG_MILKY_WAY/np.max(AVG_MILKY_WAY))*np.max(flux))

#PLOTTING -------------------------------------------------------------------

def plot_velocity_baseline(velocity,flux,title):
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
    plt.show() #Displays graph

def plot_velocity_without_baseline(velocity,flux,title):
    plt.xlabel('Velocity [kms^-1]')
    plt.ylabel('Uncalibrated Flux')
    plt.title(title+ ' Velocity-Uncorrected Flux Plot (Background Corrected)')

    #plt.ylim([-1,+1])
    plt.plot(velocity, flux,PD,zorder=0)
    plt.legend(["Data"])

    plt.show()

def plot_velocity_without_baseline_corrected(velocity,flux,title):
    plt.xlabel('Velocity [kms^-1]')
    plt.ylabel('Flux [K]')
    plt.title(title+ ' Velocity-Flux Plot (Corrected)')

    #PLOTTING DATA ---------------------------------------
    #plt.ylim([-1,+1])
    plt.plot(velocity, flux,PD,zorder=0)
    plt.legend(["Data"])

    plt.show()

def plot_coordinates():
    ra_coords = []
    dec_coords = []
    titles = []
    for entry in os.scandir("M31Backup\\"):
        title = entry.path[10:-4]
        if title.startswith("M"):
            ra_coords.append(FILE_COORDS[title][0])
            dec_coords.append(FILE_COORDS[title][1])

    plt.xlabel('Right Ascension [degrees]')
    plt.ylabel('Declination [degrees]')
    plt.title("Coordinates")
    plt.plot(ra_coords,dec_coords,"b.")
    plt.gca().invert_xaxis()
    for i,text in enumerate(titles):
        plt.annotate(text,(ra_coords[i],dec_coords[i]))
    plt.show()

#MAIN FUNCTIONS ------------------------------------------------------------

def calibrate(path,rtn=False,plot=False):
    title = path[10:-4]
    metadata = np.genfromtxt(path,delimiter='=',skip_header=0, skip_footer=514,dtype=str)
    velocity_addition = float(metadata[-1][-1])
    bin_width = float(metadata[-5][-1])
    elevation = float(metadata[-11][-1])
    data = np.genfromtxt(path,dtype='float',delimiter='/',skip_header=27, skip_footer=1)

    scaled_velocity = np.array(data[:,1])
    uncalibrated_flux =  np.array(data[:,0])

    velocity = scaled_velocity*bin_width + velocity_addition

    #REMOVE BASELINE
    cropped_velocity,cropped_flux = remove_peaks(velocity,uncalibrated_flux,title)
    fit = fit_equation(cropped_velocity,cropped_flux)
    fit = np.polyfit(cropped_velocity,cropped_flux,deg=4)

    flux_without_baseline = uncalibrated_flux - np.polyval(fit, velocity)

    #CALIBRATE FOR ELEVATION
    elevation_calibrated_flux = calibrate_elevation(flux_without_baseline, elevation)

    #SCALE BRIGHTNESS
    calibrated_flux = elevation_calibrated_flux*FLUX_CALIBRATION

    if plot:
        #ORIGINAL
        #plot_velocity_baseline(velocity,uncalibrated_flux,title)

        #REMOVE BASELINE
        plot_velocity_without_baseline(velocity,flux_without_baseline,title)

        #CALIBRATE FOR ELEVATION
        #plot_velocity_without_baseline(velocity,elevation_calibrated_flux,title)

        #SCALE BRIGTNESS
        calibrated_flux = elevation_calibrated_flux*FLUX_CALIBRATION
        plot_velocity_without_baseline_corrected(velocity,calibrated_flux,title)

    if rtn:
        return velocity,calibrated_flux


def integrate(velocity,flux,title,rtn=False,plot=False):
    plt.plot(velocity, flux,PD,zorder=0)
    #Remove mystery peak
    flux[338:342] = (flux[342] - flux[338]) * np.array([0,1,2,3]) / (velocity[342]-velocity[338]) + flux[338]
    if int_params[title][2] == "1":
        min_bound = int(int_params[title][0])
        max_bound = int(int_params[title][1])
        if plot:
            plt.plot([velocity[min_bound],velocity[min_bound]],[0,max(flux)],"r--")
            plt.plot([velocity[max_bound],velocity[max_bound]],[0,max(flux)],"r--")
        local_corrected_flux = flux[min_bound:max_bound]
        local_corrected_velocity = velocity[min_bound:max_bound]
    if int_params[title][2] == "0":
        #local_corrected_velocity, local_corrected_flux = remove_local_hydrogen(velocity, flux, title)
        local_corrected_velocity, local_corrected_flux = remove_local_hydrogen_by_average(velocity, flux)

    if plot:
        plt.xlabel('Velocity [kms^-1]')
        plt.ylabel('Flux [K]')
        plt.title(title+ ' Velocity-Flux Plot (Corrected)')
        #plt.xlim([-150,100])
        plt.plot(local_corrected_velocity, local_corrected_flux,"r--",zorder=0)
        plt.show() #Displays graph
    area = simpson(local_corrected_flux, local_corrected_velocity)
    if rtn:
        return area


#BATCH PROCESS DATA ----------------------------------------------------------
def plot_all_graphs():
    for entry in os.scandir("M31Backup\\"):
        title = entry.path[10:-4]
        if title.startswith("M"):
            calibrate(entry.path,plot=True)

def integrate_all_graphs():
    results = []
    ra_coords = []
    dec_coords = []
    titles = []
    for entry in os.scandir("M31Backup\\"):
        title = entry.path[10:-4]
        if title.startswith("M"):
            titles.append(title)
            velocity,flux = calibrate(entry.path,rtn=True)
            area = integrate(velocity,flux,title,rtn=True)
            results.append(area)
            ra_coords.append(FILE_COORDS[title][0])
            dec_coords.append(FILE_COORDS[title][1])
    print(results)
    plt.xlabel('Right Ascension [degrees]')
    plt.ylabel('Flux [K km^s]')
    plt.title("Integration by coordinates")
    plt.plot(ra_coords,results,"rx")
    plt.gca().invert_xaxis()
    plt.show()

    plt.xlabel('Declination [degrees]')
    plt.ylabel('Flux [K km^s]')
    plt.title("Integration by coordinates")
    plt.plot(dec_coords,results,"gx")
    plt.gca().invert_xaxis()
    plt.show()

    plt.ylabel('Declination [degrees]')
    plt.xlabel('Right Ascension [degrees]')
    plt.title("Flux Integration contour")
    plt.gca().invert_xaxis()

    z = np.zeros((9,12))

    for i,title in enumerate(titles):
        title_nums = title[4:]
        y_index = int(title_nums[:-1])-1
        x_index = int(title_nums[-1])
        #print(x_index,y_index)
        z[x_index][y_index] =  results[i]
        #z[x_index][y_index] =  title_nums #

    # for i in range(0,9):
    #     print(z[i])
    z = np.transpose(z)
    #Z is in correct format, 9 x 12
    x = np.unique(ra_coords)
    print(x) #length 9
    y = np.unique(dec_coords)
    print(y) #length 12
    X,Y = np.meshgrid(x,y) # 12 x 9
    print(np.shape(X))
    # plt.xlim(0.5,0.3)
    # plt.ylim(39,43)

    # plt.ylim(0.6,0.3) #wrong
    # plt.xlim(39,45)
    plt.contourf(X,Y,z)
    #plt.clabel(plt.contour(X,Y,z),inline=True, fontsize=10)
    plt.colorbar()
    plt.show()


#plot_coordinates()

FLUX_CALIBRATION = calibrate_flux("M31Backup\\S8A.TXT")
FILE_COORDS = match_coords()
AVG_MILKY_WAY = get_average_milkyway()
#print(FLUX_CALIBRATION)

# individual_path = "M31Backup\\M31P18.TXT"
# velocity,flux = analyse(individual_path,True)
# plot_velocity_for_integration(velocity,flux,individual_path[10:-4])
#plot_all_graphs()
integrate_all_graphs()
