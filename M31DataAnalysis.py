#IMPORTS -------- ----------------------------------------------------------

import os
import math
import numpy as np #Import numpy for maths calculations
import matplotlib.pyplot as plt #Import pyplot for plotting the graph
from scipy.integrate import simpson
from scipy.interpolate import griddata

MAX_DEGREE = 5
S8_FLUX = 850 #K km/s
M31DIST = 752 #Kpc
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
            dec_degrees, dec_minutes = metadata[6][-1].split(".")
            dec = float(dec_degrees) + (float(dec_minutes[:2])/60)
            ra_hours,ra_mins = metadata[5][-1].split(".")
            ra = (float(ra_hours) + float(ra_mins[:2])/60)*15#np.cos(np.deg2rad(dec))

            file_coordinates[title] = [ra,dec]
    return file_coordinates #Returns dictionary

# DEFINE MATHS FUNCTIONS --------------------------------------------------

def root_mean_square(x,y,fit):
    mean_sq = np.sqrt(np.sum(np.abs(y-np.polyval(fit, x))**2)/np.size(y))
    return mean_sq

def fit_equation(x,y,max_deg=MAX_DEGREE): #Fits a poly equation to data
    best_fit = 1
    best_res = 10000000
    for degree in range(1,max_deg):
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
    #print(area)
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
    #boolean_to_remove[5] = False
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

#MASS OF HYDROGEN ----------------------------------------------------------

def get_neutral_H_mass_density(flux,velocity):
    frequency = velocity #*4.762e3
    delta_f = np.average(np.diff(frequency))
    integral = delta_f*np.sum(flux)
    integral = (frequency[1]-frequency[0])*np.sum(flux)
    # integral_s = simpson(flux,frequency)
    # mass_atoms_cm = 3.488e14*integral
    # mass_sol_cm = 8.396e-58*mass_atoms_cm
    # mass_sol_kpc = mass_sol_cm*1/(1.05e-43)
    # mass_sol = mass_sol_kpc*4.83
    #print(4.762e3*3.488e14*8.396e-58*(1/1.05e-43)) #Should be 14604
    mass_sol = 14604*integral
    return mass_sol

def get_neutral_H_mass(mass_density,title):
    dec = float(FILE_COORDS[title][1])
    area = np.deg2rad(0.5)*np.deg2rad(0.33)*np.cos(np.deg2rad(dec))*M31DIST**2
    mass = mass_density*area#square_rectangle_kpc
    return mass

def get_M31_area(ra_coords,dec_coords):
    area = 0
    for dec in dec_coords:
        area += np.deg2rad(0.5)*np.deg2rad(0.33)*np.cos(np.deg2rad(dec))*M31DIST**2
    return area

#PLOTTING -------------------------------------------------------------------

def plot_velocity_baseline(velocity,flux,title):
    plt.xlabel('Velocity [kms^-1]')
    plt.ylabel('Uncalibrated Flux')
    plt.title(title+ ' Velocity-Uncorrected Flux Plot')

    #PLOTTING DATA -----------------

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
    fig, ax = plt.subplots()
    for entry in os.scandir("M31Backup\\"):
        title = entry.path[10:-4]
        if title.startswith("M"):
            ra_coords.append(FILE_COORDS[title][0])
            dec_coords.append(FILE_COORDS[title][1])
            titles.append(title[4:])

    plt.xlabel('Right Ascension [degrees]')
    plt.ylabel('Declination [degrees]')
    plt.title("Coordinates")
    ax.plot(ra_coords,dec_coords,"r.")
    plt.gca().invert_xaxis()
    #beam width is 11.6 arcsec
    ax.set_aspect('equal', adjustable='datalim')
    radius_deg = (11.6/60)/2
    plot_shapes = False
    for i,text in enumerate(titles):
        ax.annotate(text,(ra_coords[i],dec_coords[i]))
        if plot_shapes:
            circle1 = plt.Circle((ra_coords[i],dec_coords[i]), radius_deg, color='b', fill=False)
            ax.add_patch(circle1)
            height = 0.5#*np.cos(np.deg2rad(dec_coords[i]))
            width = 0.33
            rectangle1 = plt.Rectangle((ra_coords[i]-height/2,dec_coords[i]-width/2),height,width,color='b', fill=False)
            ax.add_patch(rectangle1)
    #plt.show()

def plot_mass_contour(ra_coords,dec_coords,results,titles):
    plt.gca().invert_xaxis()
    plt.ylabel('Declination [degrees]')
    plt.xlabel('Right Ascension [degrees]')
    plt.title("Neutral HI mass density contour")

    z = np.zeros((9,12))
    for i,title in enumerate(titles):
        title_nums = title[4:]
        x_index = int(title_nums[-1])
        y_index = int(title_nums[:-1])-1
        z[x_index][y_index] =  results[i]

    z = np.transpose(z)
    x = np.unique(ra_coords)
    y = np.unique(dec_coords)
    x = np.flip(x)
    levels=np.linspace(min(results),max(results),15)
    X,Y = np.meshgrid(x,y) # 12 x 9
    #plt.xlim(max(x),8.4)
    #plt.ylim(38.5,42.7)
    plt.contourf(X,Y,z,levels=levels)
    plt.colorbar(label="")
    plt.show()

def oliver_contour(X,Y,z):
    #First i generate a higher resolution dataset.
    #I do this to make it easy to generate a straight line cross section
    z_straight = np.matrix.flatten(z)
    points = np.transpose(np.vstack((np.matrix.flatten(X),np.matrix.flatten(Y))))
    grid_x, grid_y = np.mgrid[12:8:100j, 39:42.5:100j]
    z_interpolate = griddata(points, z_straight, (grid_x, grid_y), method='linear')


    #These arrays define the line to sample along
    #Tinker the start and end indexes to change the direction of the line
    x_index = np.linspace(8,97,12)
    y_index = np.linspace(99,1, 12)

    #initializing arrays that will contain position and speed values
    speed_vals = np.array([])
    pos_vals = np.zeros(x_index.shape)

    #These lines grab the values along the line. note how i round the indexes.
    speed_vals = z_interpolate[np.round(y_index).astype(int),np.round(x_index).astype(int)]#-mean_redshift
    RA_vals = grid_x[np.round(y_index).astype(int),np.round(x_index).astype(int)]
    DEC_vals = grid_y[np.round(y_index).astype(int),np.round(x_index).astype(int)]

    #This plots the line i have sampled along. This line is only straight as I
    #Have interpolated a higher resolution dataset
    plt.plot(RA_vals, DEC_vals, "r")

    #This is handy for doing a line integral across the galaxy
    RA_diff = np.deg2rad(np.diff(RA_vals))
    DEC_diff = np.deg2rad(np.diff(DEC_vals))

    #distance of andromeda
    R = 752

    # RA_rel = X - center_RA
    # DEC_rel = Y - center_DEC

    # position_vec_DEC = R*np.deg2rad(DEC_rel)
    # position_vec_RA = R*np.sin(-np.deg2rad(Y)+np.pi/2)*np.deg2rad(RA_rel)

    # position_vec = np.dstack((position_vec_RA, position_vec_DEC))

    # rotation_matrix = np.array([[np.cos(-alpha), -np.sin(-alpha)],
    #                             [np.sin(-alpha), np.cos(-alpha)]])

    # theta = np.empty((position_vec.shape[0], position_vec.shape[1]))

    # for i in range(position_vec.shape[0]):
    #     for j in range(position_vec.shape[1]):
    #         position_vec[i,j,:] = np.matmul(rotation_matrix, position_vec[i,j,:])
    #         theta[i,j] = np.arctan2(position_vec[i,j,1],position_vec[i,j,0])
    #         theta[i,j] += np.pi

    #This for loop does a line inetgral across the galaxy to work out the position
    #Of the different points in kPc
    for i in range(RA_diff.size):
        delta_r = R*np.sqrt(DEC_diff[i]**2 + np.sin(-np.deg2rad(DEC_vals[i+1])+np.pi/2)**2*RA_diff[i]**2)
        pos_vals[i+1]= pos_vals[i]+delta_r

    #This is the datapoint at which i flip the data, this could use tweaking.w
    flip_rad = pos_vals[6]
    flip_speed = speed_vals[6]

    #This code simply plots what i have taken to be the center
    #plt.plot(grid_x[np.round(y_index[6]).astype(int),np.round(x_index[6]).astype(int)],
    #             grid_y[np.round(y_index[6]).astype(int),np.round(x_index[6]).astype(int)], "bx")


    #Here i flip the data around the center, creating a galactic rotation curve
    upper_index = np.where(pos_vals>=flip_rad)
    lower_index = np.where(pos_vals<flip_rad)

    pos_vals[upper_index] -= flip_rad
    speed_vals[upper_index] -= flip_speed

    pos_vals[lower_index] = -pos_vals[lower_index] + flip_rad
    speed_vals[lower_index] = -speed_vals[lower_index] + flip_speed

    #Plot rotation curve

    plt.plot(pos_vals, speed_vals, 'r.')
    plt.ylabel("sampled values (km/s)")
    plt.xlabel("Distance from center (kPc)")
    plt.show()

# def get_all_r_thetas(ra_coords, dec_coords,M31_centre,M31_tilt):

#     centre_RA, centre_DEC = M31_centre
#     RA_rel = ra_coords - centre_RA
#     DEC_rel = dec_coords - centre_DEC

#     position_vec_DEC = M31DIST*np.deg2rad(DEC_rel)
#     position_vec_RA = M31DIST*np.sin(-np.deg2rad(centre_DEC)+np.pi/2)*np.deg2rad(RA_rel)

#     position_vec = np.dstack((position_vec_RA, position_vec_DEC))

#     rotation_matrix = np.array([[np.cos(-M31_tilt), -np.sin(-M31_tilt)],
#                             [np.sin(-M31_tilt), np.cos(-M31_tilt)]])

#     theta = np.empty((position_vec.shape[0], position_vec.shape[1]))

#     for i in range(position_vec.shape[0]):
#         for j in range(position_vec.shape[1]):
#             position_vec[i,j,:] = np.matmul(rotation_matrix, position_vec[i,j,:])
#             theta[i,j] = np.arctan2(position_vec[i,j,1],position_vec[i,j,0])
#             theta[i,j] += np.pi
#     return position_vec, theta

def get_error_on_mean_speed(flux, velocity):
    flux[flux<0] = 0
    I1 = np.sum(flux*velocity)
    I2 = np.sum(flux)
    I3 = np.sum(flux*velocity**2)
    
    variance = I3/I2 - (I1/I2)**2
    #print(variance)
    error = np.sqrt(variance)
    #print(error)
    return error

def plot_velocity_contour(ra_coords,dec_coords,results,results_errors, titles):
    #Do subtract bulk motion (speed of centre, taken to be 0) and take absolute value
    #Convert distance to centre into kpc not degrees

    M31_centre = (9.9,40.75)
    #Random functions ----------------------------

    def grad(p1,p2):
        x1,y1 = p1
        x2,y2 = p2
        return(y2-y1)/(x2-x1)

    def angle(grad1,grad2):
        return np.arctan((grad2-grad1)/(1+grad1*grad2))

    def dist_from_centre(x1,y1):
        #centre at 9.9,40.75
        #xc,yc = 10.6848333,41.2689028 # from https://esahubble.org/images/opo1204b/
        xc,yc = M31_centre
        ang_dist = np.sqrt((xc-x1)**2+(yc-y1)**2) #in degrees
        return M31DIST*np.deg2rad(ang_dist)

    def dist_from_point(x1,y1,x2,y2):
        #centre at 9.9,40.75
        return np.sqrt((x2-x1)**2+(y2-y1)**2)

    M31_major_grad = grad(FILE_COORDS["M31P120"],FILE_COORDS["M31P18"])
    M31_bulk_motion = -297 #results_dict["M31P64"]
    M31_semi_major_axis_length = dist_from_point(*FILE_COORDS["M31P120"],*FILE_COORDS["M31P18"])
    M31_semi_minor_axis_length = dist_from_point(*FILE_COORDS["M31P86"],*FILE_COORDS["M31P53"])
    M31_inc = np.rad2deg(np.arccos(M31_semi_minor_axis_length/M31_semi_major_axis_length))
    M31_tilt = np.rad2deg(angle(0,M31_major_grad))
    M31_tilt = np.rad2deg(0.833)  #FIX
    print("M31 Major Axis Gradient: ", M31_major_grad)
    print("M31 Semi-Major Axis (degrees): ",M31_semi_major_axis_length)
    print("M31 Semi-Major Axis (degrees): ",M31_semi_minor_axis_length)
    print("M31 Inclination: ", M31_inc)
    print("M31 Semi-major axis tilt: ", M31_tilt)
    print("M31 Bulk motion (centre speed): ", M31_bulk_motion)
    M31_tilt = np.rad2deg(0.833)

    def get_x_y(r_coord, dec_coord):
        #Convert from degrees to kpc coordinate system
        #centred at (centre ra, centre dec)
        x_kpc_radec = M31DIST*np.deg2rad(r_coord-M31_centre[0])
        y_kpc_radec = M31DIST*np.deg2rad(dec_coord-M31_centre[1])

        #Convert from ra dec axis to major/minor axis
        theta = -np.deg2rad(M31_tilt) #might neeed to be 90- or 360-
        x_kpc_semis = x_kpc_radec*np.cos(theta) -y_kpc_radec*np.sin(theta)
        y_kpc_semis = x_kpc_radec*np.cos(theta) +y_kpc_radec*np.sin(theta)

        #Convert from observed distances to actual distances
        #taking into account inclination
        x_plane = x_kpc_semis
        y_plane = y_kpc_semis/np.sin(np.deg2rad(M31_inc))
        return x_plane,y_plane

    def get_r_theta(r_coord, dec_coord):
        #Convert from degrees to kpc coordinate system
        #centred at (centre ra, centre dec)
        x_plane, y_plane = get_x_y(r_coord, dec_coord)
        dist_to_centre = np.sqrt(x_plane**2+y_plane**2)
        angle_to_semi_major = np.abs(np.rad2deg(np.arctan(y_plane/x_plane)))
        print(angle_to_semi_major)
        return dist_to_centre, angle_to_semi_major

    plot_deprojection = False
    if plot_deprojection:
        #PLot coordinates in ra, dec
        x,y = ra_coords , dec_coords
        fig, ax = plt.subplots()
        ax.plot(x,y,"x")
        ax.plot([min(x),max(x)],[min(y),max(y)],"r-")
        for i,text in enumerate(titles):
            ax.annotate(text[4:],(x[i],y[i]))
        plt.xlabel("RA (degrees)")
        plt.ylabel("DEC (degrees)")
        plt.show()

        #Plot coordinates in dx, dy
        x,y = M31DIST*np.deg2rad(np.array(ra_coords)-M31_centre[0]) , M31DIST*np.deg2rad(np.array(dec_coords)-M31_centre[1])
        fig, ax = plt.subplots()
        ax.plot(x,y,"x")
        ax.plot([min(x),max(x)],[min(y),max(y)],"r-")
        for i,text in enumerate(titles):
            ax.annotate(text[4:],(x[i],y[i]))
        plt.xlabel("X Distance from centre (kpc)")
        plt.ylabel("Y Distance from centre (kpc)")
        plt.show()

        #Plot deprojected coordinates
        x,y = get_x_y(np.array(ra_coords), np.array(dec_coords))
        fig, ax = plt.subplots()
        ax.plot(x,y,"x")
        ax.plot([min(x),max(x)],[0,0],"r-")
        for i,text in enumerate(titles):
            ax.annotate(text[4:],(x[i],y[i]))
        plt.xlabel("X Distance from semi-minor axis (kpc)")
        plt.ylabel("Y Distance from semi-major axis (kpc)")
        plt.show()

        #Plot deprojected angles
        x,y = get_x_y(np.array(ra_coords), np.array(dec_coords))
        fig, ax = plt.subplots()
        ax.plot(x,y,"x")
        ax.plot([min(x),max(x)],[0,0],"r-")
        for i,text in enumerate(titles):
            ax.annotate(text[4:],(x[i],y[i]))
            ax.plot([0,x[i]],[0,y[i]],"-")
        plt.xlabel("X Distance from semi-major axis (kpc)")
        plt.ylabel("Y Distance from semi-minor axis (kpc)")
        plt.show()

        #Plot radii and distances
        rs,thetas = get_r_theta(np.array(ra_coords), np.array(dec_coords))
        fig, ax = plt.subplots()
        ax.plot(rs,thetas,"x")
        #ax.plot([0,max(x)],[0,0])
        for i,text in enumerate(titles):
            ax.annotate(text[4:],(rs[i],thetas[i]))
        plt.xlabel("Distance from centre (kpc)")
        plt.ylabel("Angle from semi-major axis (degrees)")
        plt.show()

        #Plot cos radii and distances
        rs,thetas = get_r_theta(np.array(ra_coords), np.array(dec_coords))
        fig, ax = plt.subplots()
        ax.plot(rs,np.cos(np.deg2rad(thetas)),"x")
        #ax.plot([0,max(x)],[0,0])
        for i,text in enumerate(titles):
            ax.annotate(text[4:],(rs[i],np.cos(np.deg2rad(thetas[i]))))
        plt.xlabel("Distance from centre (kpc)")
        plt.ylabel("Cosine of Angle from semi-major axis")
        plt.show()


    #Plot velocity contour ----------------------
    #plot_coordinates()
    #plt.gca().invert_xaxis()

    plt.ylabel('Declination [degrees]')
    plt.xlabel('Right Ascension [degrees]')
    plt.title("Velocity contour")

    results_dict = {}
    errors_dict = {}
    z = np.zeros((9,12))
    for i,title in enumerate(titles):
        title_nums = title[4:]
        x_index = int(title_nums[-1])
        y_index = int(title_nums[:-1])-1
        z[x_index][y_index] =  results[i]
        results_dict[title] = results[i]
        errors_dict[title] = results_errors[i]

    z = np.transpose(z)
    x = np.unique(ra_coords)
    y = np.unique(dec_coords)
    #x,y = get_x_y(x,y)
    #x = M31DIST*np.deg2rad(x-M31_centre[0])
    #y = M31DIST*np.deg2rad(y-M31_centre[1])

    #x = x*np.cos(np.deg2rad(M31_tilt)) -y*np.sin(np.deg2rad(M31_tilt))
    #y = x*np.cos(np.deg2rad(M31_tilt)) +y*np.sin(np.deg2rad(M31_tilt))

    x = np.flip(x)

    #NOTE - M31 bulk motion should be -297 ish not -381
    M31_major_axis = M31_major_grad*x+(np.min(y)-2*(np.max(y)-np.min(y)))
    M31_minor_axis = -M31_major_grad*x+(np.max(y)+2*(np.max(y)-np.min(y)))
    #plt.plot(x,M31_major_axis,"r-")
    #plt.plot(x,M31_minor_axis,"b-")

    levels=np.linspace(min(results),max(results),30)
    #levels = np.linspace(-800,200,20)
    X,Y = np.meshgrid(x,y) # 12 x 9
    plt.contour(X,Y,z,levels=levels)
    plt.colorbar()
    #plt.gca().set_aspect('equal')
    plt.show()


    #oliver_contour(X, Y, z)

    #Simple diagonal method for velocity graph -----------------------------
    diagonal_titles = [120,111,92,83,74,65,56,47,28]
    speed_vals = np.array([])
    speed_errors = np.array([])
    pos_vals = np.array([])
    
    for t in diagonal_titles:
        r,semi_major_angle = get_r_theta(*FILE_COORDS["M31P"+str(t)])
        #print(r,semi_major_angle)
        point_speed = results_dict["M31P"+str(t)]
        
        point_speed = np.abs(point_speed-M31_bulk_motion)
        actual_speed = point_speed/(np.sin(np.deg2rad(M31_inc))*np.cos(np.deg2rad(semi_major_angle)))
        point_error = errors_dict["M31P"+str(t)]
        actual_error =(point_error/point_speed)*actual_speed
        speed_vals = np.append(speed_vals, actual_speed)
        speed_errors = np.append(speed_errors, actual_error)
        pos_vals = np.append(pos_vals, r)

    plt.errorbar(pos_vals[:5],speed_vals[:5],yerr=speed_errors[:5],fmt="rx")
    plt.errorbar(pos_vals[5:],speed_vals[5:],yerr=speed_errors[5:],fmt="bx")
    plt.legend(["Upper Left Quadrant", "Lower Right Quadrant"])
    plt.title("Velocity curve (method 1, taking diagonals)")
    plt.xlabel("Distance from M31 centre (kpc)")
    plt.ylabel("Velocity (km/s)")
    plt.show()

    #My method for velocity graph --------------------
    speed_vals = np.array([])
    pos_vals = np.array([])
    speed_errors = np.array([])

    for i,title in enumerate(titles):
        title_nums = title[4:]
        x = int(title_nums[-1])
        y = int(title_nums[:-1])-1

        r,semi_major_angle = get_r_theta(*FILE_COORDS[title])
        point_speed = results_dict[title]
        point_speed = np.abs(point_speed-M31_bulk_motion)
        actual_speed = point_speed/(np.sin(np.deg2rad(M31_inc))*np.cos(np.deg2rad(semi_major_angle)))
        speed_vals = np.append(speed_vals, actual_speed)
        
        point_error = errors_dict["M31P"+str(t)]
        actual_error =(point_error/point_speed)*actual_speed
        speed_errors = np.append(speed_errors, actual_error)
        
        pos_vals = np.append(pos_vals, r)

    #plt.plot(pos_vals,speed_vals*-1,"rx")
    plt.errorbar(pos_vals[speed_vals<500],speed_vals[speed_vals<500],yerr=speed_errors[speed_vals<500],fmt="rx")
    #plt.plot(x_index,y_index)
    #print(speed_vals)
    #print(pos_vals)
    plt.title("Velocity curve (method 3, looking at every point)")
    plt.xlabel("Distance from M31 centre (kpc)")
    plt.ylabel("Velocity (km/s)")
    plt.show()


    """
    #Interpolate method for contours ---------------------------------------
    inter_grid = griddata((ra_coords,dec_coords), results, (X,Y), method='cubic')
    plt.gca().invert_xaxis()
    plt.ylabel('Declination [degrees]')
    plt.xlabel('Right Ascension [degrees]')
    plt.title("Velocity contour")
    levels=np.linspace(min(results),max(results),30)
    plt.contour(X,Y,inter_grid,levels=levels)
    plt.colorbar()
    plt.show()
    #print(inter_grid)
    #print(np.diag(inter_grid))
    speed_vals = np.array([])
    pos_vals = np.array([])
    for i,title in enumerate(titles):
      title_nums = title[4:]
      x = int(title_nums[-1])
      y = int(title_nums[:-1])-1
      point_speed = z[y,x] #(z[y,x]+z[y+1,x+1]+z[y,x+1]+z[y+1,x])/4
      #title = "M31P"+ str(y+1)+str(x+1)
      point_grad = grad([9.9,40.75],FILE_COORDS[title] )
      semi_major_angle = angle(M31_major_grad,point_grad )
      actual_speed = point_speed/(np.sin(np.deg2rad(M31_inc))*np.cos(np.deg2rad(semi_major_angle)))
      speed_vals = np.append(speed_vals, actual_speed)
      #speed_vals = np.append(speed_vals, point_speed)
      #pos_vals = np.append(pos_vals, np.sqrt((12-X[y,x])**2+(39-Y[y,x])**2))
      #pos_vals = np.append(pos_vals, dist_from_centre(X[y,x],Y[y,x]))
      pos_vals = np.append(pos_vals, dist_from_centre(*FILE_COORDS[title]))

    #plt.plot(pos_vals,speed_vals*-1,"rx")
    plt.plot(pos_vals*deg_to_kpc,np.abs(speed_vals-M31_bulk_motion),"rx")
    plt.title("Velocity curve (method 4, looking at every point interpolation)")
    plt.xlabel("Distance from M31 centre (kpc)")
    plt.ylabel("Velocity (km/s)")
    plt.show()
    """


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
        plot_velocity_baseline(velocity,uncalibrated_flux,title)

        #REMOVE BASELINE
        #plot_velocity_without_baseline(velocity,flux_without_baseline,title)

        #CALIBRATE FOR ELEVATION
        #plot_velocity_without_baseline(velocity,elevation_calibrated_flux,title)

        #SCALE BRIGTNESS
        calibrated_flux = elevation_calibrated_flux*FLUX_CALIBRATION
        plot_velocity_without_baseline_corrected(velocity,calibrated_flux,title)

    if rtn:
        return velocity,calibrated_flux

def integrate(velocity,flux,title,rtn=False,plot=False):
    #Remove mystery peak
    flux[338:342] = (flux[342] - flux[338]) * np.array([0,1,2,3]) / (velocity[342]-velocity[338]) + flux[338]

    min_bound = int(int_params[title][0])
    max_bound = int(int_params[title][1])

    #local_corrected_velocity, local_corrected_flux = remove_local_hydrogen(velocity, flux, title)
    local_corrected_velocity, local_corrected_flux = remove_local_hydrogen_by_average(velocity, flux)

    local_corrected_flux = local_corrected_flux[min_bound:max_bound]
    local_corrected_velocity = local_corrected_velocity[min_bound:max_bound]

    if plot:
        plt.plot(velocity, flux,PD,zorder=0)
        plt.plot([velocity[min_bound],velocity[min_bound]],[0,max(flux)],"r--")
        plt.plot([velocity[max_bound],velocity[max_bound]],[0,max(flux)],"r--")
        plt.xlabel('Velocity [kms^-1]')
        plt.ylabel('Flux [K]')
        plt.title(title+ ' Velocity-Flux Plot (Corrected)')
        plt.plot(local_corrected_velocity, local_corrected_flux,"r--",zorder=0)
        plt.show()

    area = simpson(local_corrected_flux, local_corrected_velocity)
    delta_v = np.average(np.diff(local_corrected_velocity))
    zeroth_moment = delta_v*np.sum(local_corrected_flux)
    mass_density = get_neutral_H_mass_density(local_corrected_flux, local_corrected_velocity)
    #local_corrected_flux[local_corrected_flux<0] = 0 #BODGE
    first_moment = np.sum(local_corrected_velocity*local_corrected_flux)/np.sum(local_corrected_flux)
    first_moment_error = get_error_on_mean_speed(local_corrected_flux, local_corrected_velocity)
    #print(first_moment, first_moment_error)
    if rtn:
        return area, mass_density, zeroth_moment, first_moment, first_moment_error


#BATCH PROCESS DATA ----------------------------------------------------------
def plot_all_graphs():
    for entry in os.scandir("M31Backup\\"):
        title = entry.path[10:-4]
        if title.startswith("M"):
            calibrate(entry.path,plot=True)

def integrate_all_graphs(plot=False):
    ra_coords = []
    dec_coords = []
    areas = []
    titles = []
    mass_densities = []
    zeroth_moments = []
    first_moments = []
    first_moment_errors = []
    total_mass = 0
    for entry in os.scandir("M31Backup\\"):
        title = entry.path[10:-4]
        if title.startswith("M"):
            titles.append(title)
            velocity, flux = calibrate(entry.path,rtn=True)
            area, mass_density,zeroth_moment, first_moment, first_moment_error = integrate(velocity,flux,title,rtn=True)
            #if mass_density<0:
            #    print(title, mass_density)
            total_mass += get_neutral_H_mass(mass_density, title)
            areas.append(area)
            mass_densities.append(mass_density)
            first_moments.append(first_moment)
            first_moment_errors.append(first_moment_error)
            zeroth_moments.append(zeroth_moment)
            ra_coords.append(FILE_COORDS[title][0])
            dec_coords.append(FILE_COORDS[title][1])


    if False:
        radius_kpc = (3.37e-3)/2 *M31DIST
        scan_area = np.pi*radius_kpc**2
        total_scan_area = np.pi*radius_kpc**2*len(titles)
        total_M31_area = get_M31_area(ra_coords, dec_coords)
        avg_zeroth_moment = np.average(zeroth_moments)
        avg_mass_density = np.average(mass_densities)

        mass_1  =14604*avg_zeroth_moment*total_M31_area
        #mass_2 = avg_mass_density*total_M31_area
        mass_3 = total_mass
        print("Observed area for one scan is {0:3.2f} kpc^2".format(scan_area))
        print("Total observed area for all scans is {0:3.2e} kpc^2".format(total_scan_area))
        print("Estimated M31 area by boxes is {0:3.2f} kpc^2".format(total_M31_area))
        print("The average zeroth moment is {0:3.2f} K km/s".format(avg_zeroth_moment))
        print("The average mass density is {0:3.2e} solar masses per kpc^2".format(avg_mass_density))
        print("The total mass, using average zeroth moment times area, is {0:3.2e} solar masses".format(mass_1))
        #print("The total mass, using average mass density times area, is {0:3.2e} solar masses".format(mass_2))
        print("The total mass, scaling each scan to a box, is {0:3.2e} solar masses".format(mass_3))
        #print(results)

    if plot:
        #plot_contour(ra_coords,dec_coords,areas,titles,"Flux")
        #plot_mass_contour(ra_coords,dec_coords,mass_densities,titles)
        plot_velocity_contour(ra_coords,dec_coords,first_moments,first_moment_errors,titles)

def run_for_individual(title):
    velocity,flux = calibrate("M31Backup\\"+title+".TXT",rtn=True,plot=True)
    area, mass,zeroth_moment,first_moment = integrate(velocity,flux,title,rtn=True,plot=True)
    print(title)
    print("Area:  {0:3.2f} \n Mass: {1:3.2f} \n First Moment: {2:3.2f}".format(area, mass,first_moment))

FLUX_CALIBRATION = calibrate_flux("M31Backup\\S8A.TXT")
#print(FLUX_CALIBRATION)
FILE_COORDS = match_coords()
AVG_MILKY_WAY = get_average_milkyway()

#plot_coordinates()
#run_for_individual("M31P95")
#plot_all_graphs()
integrate_all_graphs(True)
