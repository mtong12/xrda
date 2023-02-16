import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import glob as glb
import os
from PIL import Image

#These are user parameters that you change based off of what material you will be analyzing
USER_PARAMS = {
    "PEAKS TO ANALYZE" : ["pos0", "pos1", "pos2"], #What peaks to analyze example: ["pos0", "pos1", "pos2", "pos5"] corresponds with peak 1, peak 2, peak 3 and peak 6.
    "EXPERIMENT TIME" : 100, #The total amount of time the experiment takes place over (used to calculate time resolution) if exporting pressure vs time plot
    "XRAY WAVELENGTH" : 0.4066, #Wavelength of XRAY in Angstroms
    "A" : 4.03892966, #Lattice parameters of the crystal in Angstroms
    "B" : 4.03892966,
    "C" : 4.03892966,
    "BULK MODULUS" : 76.0, #Bulk modulus of the material in GPa
    "SYMMETRY" : "CUBIC", #What crystal system the material is (used to calculate lattice parameter)
    "HEATMAP DIM" : -1, #Size of X axis for heatmap. The Y dimension is automatically calculated by dividing total size of data set by X. Set this to -1 if the image is square.
    "INVALIDATE HEATMAP NEGATIVES": True, #Toggles if negative values in the heatmap are to be erased (replaces the pixel with a white square).
    "XDI MASK" : True #Toggles the image mask, "mask.png" for XDI outputs
}

#These are toggles for what to output. Change the values to True/False. Note: XDI and Dynamic outputs are selected by what folder the input csv files are located in.
OUTPUTS = {
    "CSV FILES" : True,

    #DYNAMIC OUTPUTS
    "PRESSURE VS. TIME PLOT" : True,
    "FWHM VS. TIME PLOT" : True,
    "LATTICE PARAMETER VS. TIME PLOT" : True,
    "LATTICE STRAIN VS. TIME PLOT" : True,
    "PRESSURE VS. 2-THETA VS. INTENSITY PLOT" : True,
    "TIME VS. 2-THETA VS. INTENSITY PLOT" : True,

    #XDI OUTPUTS
    "2D PRESSURE MAP" : True,
    "2D INTENSITY MAP" : True,
    "2D FWHM MAP" : True,
    "X SLICE": True #Toggles the output of a plot of 2d map vs X at the middle of the map 
}

#This defines the first 10 peaks in the crystal system (Feel free to add more peaks)
#In "PEAKS TO ANALYZE", the number after the first 3 characters of each peak will determine which direction to associate with. Ex: pos4 will associated with [2,2,2]. pos0 will associate with [1,1,1].
#The N of 'posN' is used to access the index of the array and correlate a peak with its corresponding direction.
SYMMETRY = {
    "CUBIC" : [[1,1,1],[2,0,0],[2,2,0],[3,1,1],[2,2,2],[4,0,0],[3,3,1],[4,2,0],[5,1,1],[4,4,0],[5,3,1],[6,0,0],[6,2,0],[5,3,3],[6,2,2],[4,4,4],[7,1,1],[6,4,0],[6,4,2],[7,3,1]],
    "CUSTOM" : [] #Define a custom symmetry. Ex: [[1, 1, 1], [2, 0, 0]]. This associates [111] with peak 1 (pos0) and [200] with peak 2 (pos1).
}

#This calculates the pressure and FWHM 
def calculate_values(df, peak, peak_name, wavelength, a, b, c, bulk_mod, hkl):

    df2 = pd.DataFrame()

    #Grabs the 2-theta value from XRD
    positions = df[peak]

    #Calculates and insert the interatomic spacing into our calculation dataframe using Bragg's law
    df2.insert(0, peak_name+'_d_Angstrom', wavelength/(2*np.sin((np.pi/180)*(positions/2))))

    #Calculates the lattice parameter from the spacing
    df2.insert(1, peak_name+'_a_Angstrom', df2[peak_name+'_d_Angstrom']*np.sqrt(hkl[0]**2 + hkl[1]**2 + hkl[2]**2))

    #Calculate and insert the pressure using bulk modulus into the dataframe
    df2.insert(2, peak_name+'_P_GPa', (((a*b*c)-(df2[peak_name+'_a_Angstrom']**3))/(df2[peak_name+'_a_Angstrom'][0]**3))*bulk_mod)

    #Invalidates negative sigma values
    df[df["sig"+peak[3:]] < 0] = np.nan

    #Calculates and insert the FWHM and converts to radians
    df2.insert(0, peak_name+'_FWHM', np.sqrt((8*np.log(2))*df["sig"+peak[3:]]**2)*(np.pi/180))

    #Calculates and inserts lattice strain
    df2.insert(1, peak_name+'_Lattice_Strain', (df2[peak_name+'_a_Angstrom'] - df2[peak_name+'_a_Angstrom'][0]) / df2[peak_name+'_a_Angstrom'][0])

    return df2

#This makes a new path if one doesn't exist
def newpath(filepath):
    if not os.path.exists(filepath):
        os.makedirs(filepath)

#Converts an 1D array into an array with shape (x,size(array)/x). This allows us to select an X and Y position and find its corresponding value. output[X][Y]
def conv_xy(xdim,array):
    ydim = np.floor(array.size/xdim).astype(int)
    output = []
    for i in range(0, ydim):
        output.append([])
        for j in range (0, xdim):

            #This invalidates negative values. Can be toggled by setting 'INVALIDATE HEATMAP NEGATIVES' in user paramssss.
            value = array[i * xdim + j]
            if value < 0 and USER_PARAMS['INVALIDATE HEATMAP NEGATIVES']:
                value = np.nan

            #Invalidates masked values
            if mask[i][j][3] > 0 and USER_PARAMS["XDI MASK"]:
                value = np.nan

            output[i].append(value)
    return output

#Generates a heatmap from an array
def heatmap(data, map_name, output_path, peak_name, filename, dimension):
    fig = plt.figure(figsize=(8,7))
    xdim = np.floor(np.sqrt(data.size)).astype(int) if dimension == -1 else dimension
    Z = conv_xy(xdim, data)

    #We have to transpose the X and Y for pcolormesh. shape[0] = x size, shape[1] = y size
    X = np.arange(np.array(Z).shape[1])
    Y = np.arange(np.array(Z).shape[0])
    plt.title(map_name)
    c = plt.pcolormesh(X,Y,Z)
    plt.colorbar(c)
    newpath(output_path)
    fig.savefig(output_path+peak_name+"_"+filename)
    plt.close()

    if OUTPUTS["X SLICE"]:
        fig = plt.figure(figsize=(8,7))
        g = np.array(Z)
        ydim = np.array(Z).shape[1]
        plt.title(map_name+" X slice about Y = {0}".format(ydim//2))
        plt.plot(g[ydim//2,:])
        fig.savefig(output_path+peak_name+"_SLICE_"+filename)
        plt.close()

def dynamic_plot(data, variable, output_path, peak_name, filename, autoscale):
    #Create a plot for pressure data vs time
    fig = plt.figure(figsize=(10,6))
    plt.title(peak_name+" {0} vs Time (s)".format(variable))

    #This sets the lower and upper bounds of the graph.
    if autoscale:
        max = data.max()*1.1 if data.max() > 0 else 0
        min = 0 if data.min() > 0 else data.min()*1.1
        plt.ylim(min, max)

    plt.xlabel("Time (s)")
    plt.ylabel(variable)

    #For the X axis, it generates a list of numbers from 0 to "EXPERIMENT TIME" spaced out by the time resolution
    #The time resolution is calculated by dividing "EXPERIMENT TIME" by how many data points there are
    plt.plot(np.arange(0,USER_PARAMS['EXPERIMENT TIME'],USER_PARAMS['EXPERIMENT TIME']/data.size), data)

    #Save the plot to outputs
    newpath(output_path)
    fig.savefig(output_path+peak_name+"_"+filename)
    plt.close()

def dynamic_waterfall(x, y, z, var_name, output_path, filename):
    fig = plt.figure(figsize=(8,7))
    scatter = plt.scatter(x, y,c=z, cmap=mpl.colormaps['plasma'], marker='d', s=75)
    cbar = plt.colorbar(scatter)
    cbar.ax.set_ylabel("Intensity")
    plt.xlabel("2 Theta")
    plt.ylabel(var_name)
    plt.title("{0} vs 2-Theta vs Insensity".format(var_name))

    #Save the plot to outputs
    newpath(output_path)
    fig.savefig(output_path+filename)
    plt.close()

if __name__ == "__main__":
    
    #Checks and sees if dynamic and XDI input folders are present
    newpath(os.getcwd() + "/inputs/dynamic")
    newpath(os.getcwd() + "/inputs/XDI")

    #Loads the XDI mask if toggled on
    if USER_PARAMS["XDI MASK"]:
        mask = np.array(Image.open("mask.png"))

    for input_type in glb.glob("./inputs/*"):

        #Determines which folder's csv files to look through
        in_path = input_type + "/*.csv"

        #Determines if XDI or Dynamic outputs are toggled based off of what folder inputs are in
        OUTPUTS["DYNAMIC"] = False
        OUTPUTS["XDI"] = False

        if input_type == "./inputs\dynamic":
            OUTPUTS["DYNAMIC"] = True

        if input_type == "./inputs\XDI":
            OUTPUTS["XDI"] = True

        #Search for csv files in the inputs folder and analyze each one
        for file in glb.glob(in_path):

            #Create a new dataframe object to store our output 
            df = pd.DataFrame()

            #Grabs the filename of the current csv file we're analyzing
            filename = file.split("\\")[2].split(".")[0]

            #Grabs the path we're going to save stuff to
            savepath = os.getcwd() + "\\outputs\\" + file.split("\\")[1]

            #Generate storage variables for 'waterfall' style diagram
            intensities = np.array([])
            thetas = np.array([])
            pressures = np.array([])

            #Calculate the pressure for each peak
            for peak in USER_PARAMS['PEAKS TO ANALYZE']:

                hkl = SYMMETRY[USER_PARAMS["SYMMETRY"]][int(peak[3:])]
                peak_name = "[{0}{1}{2}]".format(hkl[0],hkl[1],hkl[2])
                crystaldf = pd.read_csv(file)

                #Run the calculations with user params
                df2 = calculate_values(crystaldf, 
                peak, 
                peak_name,
                USER_PARAMS['XRAY WAVELENGTH'],
                USER_PARAMS['A'],
                USER_PARAMS['B'],
                USER_PARAMS['C'],
                USER_PARAMS['BULK MODULUS'],
                hkl)

                #Add values to the waterfall data
                intensities = np.append(intensities, crystaldf["int"+peak[3:]])
                thetas = np.append(thetas, crystaldf[peak])
                pressures = np.append(pressures, df2[peak_name+'_P_GPa'])

                #Merge the calculated data
                df = pd.concat([df, df2], axis=1)

                if OUTPUTS["XDI"]:
                    if OUTPUTS["2D PRESSURE MAP"]:
                        heatmap(df2[peak_name+'_P_GPa'], 
                        "{0} Pressure (GPa) Map".format(peak_name), 
                        "{0}\\plots\\{1}\\heatmaps\\pressure\\".format(savepath,filename), 
                        peak_name, 
                        filename, 
                        USER_PARAMS["HEATMAP DIM"])

                    if OUTPUTS["2D INTENSITY MAP"]:
                        heatmap(crystaldf["int"+peak[3:]], 
                        "{0} Intensity Map".format(peak_name), 
                        "{0}\\plots\\{1}\\heatmaps\\intensity\\".format(savepath,filename), 
                        peak_name, 
                        filename, 
                        USER_PARAMS["HEATMAP DIM"])

                    if OUTPUTS["2D FWHM MAP"]:
                        heatmap(df2[peak_name+'_FWHM'], 
                        "{0} FWHM (Radian) Map".format(peak_name), 
                        "{0}\\plots\\{1}\\heatmaps\\fwhm\\".format(savepath,filename), 
                        peak_name, 
                        filename, 
                        USER_PARAMS["HEATMAP DIM"])

                if OUTPUTS["DYNAMIC"]:
                    if OUTPUTS["PRESSURE VS. TIME PLOT"]:
                        dynamic_plot(df2[peak_name+'_P_GPa'],
                        "Pressure (GPa)"
                        ,"{0}\\plots\\{1}\\p_vs_t\\".format(savepath,filename),
                        peak_name,
                        filename,
                        True)

                    if OUTPUTS["FWHM VS. TIME PLOT"]:
                        dynamic_plot(df2[peak_name+'_FWHM'],
                        "FWHM (Rad)"
                        ,"{0}\\plots\\{1}\\fwhm_vs_t\\".format(savepath,filename),
                        peak_name,
                        filename,
                        True)

                    if OUTPUTS["LATTICE PARAMETER VS. TIME PLOT"]:
                        dynamic_plot(df2[peak_name+'_a_Angstrom'],
                        "Lattice Parameter (Å)"
                        ,"{0}\\plots\\{1}\\parameter_vs_t\\".format(savepath,filename),
                        peak_name,
                        filename,
                        False)
                    
                    if OUTPUTS["LATTICE STRAIN VS. TIME PLOT"]:
                        dynamic_plot(df2[peak_name+'_Lattice_Strain'],
                        "Lattice Strain"
                        ,"{0}\\plots\\{1}\\strain_vs_t\\".format(savepath,filename),
                        peak_name,
                        filename,
                        True)
            
            if OUTPUTS["DYNAMIC"]:
                if OUTPUTS['PRESSURE VS. 2-THETA VS. INTENSITY PLOT']:
                    dynamic_waterfall(thetas,
                    pressures,
                    intensities,
                    "Pressure (GPa)",
                    "{0}\\plots\\{1}\\pressure_vs_2-theta_vs_intensity\\".format(savepath,filename),
                    filename)

                if OUTPUTS['TIME VS. 2-THETA VS. INTENSITY PLOT']:
                    time = np.arange(0,USER_PARAMS['EXPERIMENT TIME'],USER_PARAMS['EXPERIMENT TIME']/crystaldf["int0"].size)
                    time = np.tile(time, len(USER_PARAMS["PEAKS TO ANALYZE"]))

                    dynamic_waterfall(thetas,
                    time,
                    intensities,
                    "Time (s)",
                    "{0}\\plots\\{1}\\time_vs_2-theta_vs_intensity\\".format(savepath,filename),
                    filename)

            if OUTPUTS["CSV FILES"]:
                #Save our merged pressure data to a csv file in outputs
                filepath = "{0}\\csv\\".format(savepath)
                newpath(filepath)
                df.to_csv(filepath+"P_Calc_"+filename+".csv")