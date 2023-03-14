#IMPORTS AND DATA ---------------------------------------------------------------------------------------------------------------

import numpy as np #Import numpy for maths calculations
import matplotlib.pyplot as plt #Import pyplot for plotting the graph
#from scipy import optimize
import os


def plot_coordinates():
    ra_coords = np.empty(0)
    dec_coords = np.empty(0)
    for entry in os.scandir("M31Backup\\"):
        title = entry.path[10:-4]
        if title.startswith("M"):
            #analyse(entry.path)
            title = entry.path[10:-4]
            metadata = np.genfromtxt(entry.path,delimiter='=',skip_header=0, skip_footer=514,dtype=str)
            ra = float(metadata[5][-1])
            dec = float(metadata[6][-1]) 
            ra_coords = np.hstack((ra_coords, ra))
            dec_coords = np.hstack((dec_coords, dec))
            
    #ra_coords = np.flip(ra_coords)
    #dec_coords = np.flip(dec_coords)
    dec_coords = dec_coords
    plt.xlabel('Right Ascension [degrees]')
    plt.ylabel('Declination [degrees]')
    plt.title("Coordinates")
    plt.plot(ra_coords,dec_coords,"b.")
    plt.show()

plot_coordinates()
