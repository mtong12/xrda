import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import glob as glb
import os
import h5py
from PIL import Image

#These are user parameters that you change based off of what material you will be analyzings
USER_PARAMS = {
    "EXPERIMENT TIME" : 100, #The total amount of time the experiment takes place over (used to calculate time resolution) if exporting pressure vs time plot
    "XRAY WAVELENGTH" : 0.4133, #Wavelength of XRAY in Angstroms
    "SYMMETRY" : "CUSTOM", #What crystal system the material is (used to calculate lattice parameter)
    "HEATMAP DIM" : -1, #Size of X axis for heatmap. The Y dimension is automatically calculated by dividing total size of data set by X. Set this to -1 if the image is square.
    "INVALIDATE HEATMAP NEGATIVES": False, #Toggles if negative values in the heatmap are to be erased (replaces the pixel with a white square).
    "XDI MASK" : True, #Toggles the image mask, "mask.png" for XDI outputs
    "FWHM THRESHOLD" : [0, 5], #Threshold at which to invalidate FWHM values, set any to -1 to disable.
    "INTENSITY THRESHOLD" : [0, 200], #Threshold at which to invalidate intensity values, set any to -1 to disable.
    "PRESSURE THRESHOLD" : [0, 5], #Threshold at which to invalidate pressure values, set any to -1 to disable.
    "LINE MASK": [5,40], #Region to plot in a Line plot (Excludes gaskets)
    "LINE STEP SIZE" : [0.6] #Step size per data sequence for line graphs (um)
}

#This defines the peak appearance
#In "PEAKS TO ANALYZE", the number after the first 3 characters of each peak will determine which direction to associate with. Ex: pos4 will associated with [2,2,2]. pos0 will associate with [1,1,1].
#The N of 'posN' is used to access the index of the array and correlate a peak with its corresponding direction.
SYMMETRY = {
    "FM-3M" : [[1,1,1],[2,0,0],[2,2,0],[3,1,1],[2,2,2],[4,0,0],[3,3,1],[4,2,0],[5,1,1],[4,4,0],[5,3,1],[6,0,0],[6,2,0],[5,3,3],[6,2,2],[4,4,4],[7,1,1],[6,4,0],[6,4,2],[7,3,1]],
    "CUSTOM" : [[1,0,2],[1,1,-1],[1,1,1]] #Define a custom symmetry. Ex: [[1, 1, 1], [2, 0, 0]]. This associates [111] with peak 1 (pos0) and [200] with peak 2 (pos1).
}

#Defines what materials are in the system. Formatted as [[(Peak locations)], [A, B, C, Bulk Modulus]]
MATERIALS = {
#    "Al" : [["pos1","pos0"], [4.0389, 4.0389, 4.0389, 76.0]] #Powder

#    "Al" : [["pos1","pos0"], [4.0509, 4.0509, 4.0509, 76.0]] #Sheet

#    "Cu" : [["pos0","pos2"],[3.6173, 3.6173, 3.6173, 139.0]], 
#    "Ni" : [["pos1","pos3"],[3.5220, 3.5220, 3.5220, 162]],

#    "Ni" : [["pos0","pos1"],[3.5240, 3.5240, 3.5240, 162]],

#    "Fe" : [["pos0","pos1","pos2"],[2.8696, 2.8696, 2.8696, 170]],
#    "Mn" : [["pos0","pos1","pos2"],[8.6185, 8.6185, 8.6185, 120]],

#    "Fe" : [["pos0","pos1","pos2"],[2.8696, 2.8696, 2.8696, 170]]

#    "Cu" : [["pos0","pos1"],[3.6173, 3.6173, 3.6173, 139.0]], 

    "Bi" : [["pos0","pos1"],[4.5470, 4.5470, 11.8571, 36.0]],
    "Cu" : [["pos2"],[3.6150, 3.6150, 3.6150, 140.0]],
}

#These are toggles for what to output. Change the values to True/False. Note: XDI and Dynamic outputs are selected by what folder the input csv files are located in.
OUTPUTS = {
    "CSV FILES" : True,

    #DYNAMIC OUTPUTS
    "PRESSURE VS. TIME PLOT" : True,
    "INTENSITY VS. TIME PLOT": True,
    "FWHM VS. TIME PLOT" : True,
    "LATTICE PARAMETER VS. TIME PLOT" : True,
    "LATTICE STRAIN VS. TIME PLOT" : True,
    "PRESSURE VS. 2-THETA VS. INTENSITY PLOT" : True,
    "TIME VS. 2-THETA VS. INTENSITY PLOT" : True,

    #XDI OUTPUTS
    "2D PRESSURE MAP" : True,
    "2D INTENSITY MAP" : True,
    "2D FWHM MAP" : True,
    "X SLICE": True, #Toggles the output of a plot of 2d map vs X at the middle of the map 

    #LINE OUTPUTS
    "LINE PRESSURE VS. TIME PLOT" : True,
    "LINE INTENSITY VS. TIME PLOT": True,
    "LINE FWHM VS. TIME PLOT" : True,
    "LINE LATTICE PARAMETER VS. TIME PLOT" : True,
    "LINE LATTICE STRAIN VS. TIME PLOT" : True,
}

#This calculates the pressure and FWHM 
def calculate_values(df, peak, peak_name, wavelength, a, b, c, bulk_mod, hkl):

    df2 = pd.DataFrame()

    #Grabs the 2-theta value from XRD
    positions = df[peak]

    #Calculates and insert the interatomic spacing into our calculation dataframe using Bragg's law
    df2.insert(0, peak_name+'_d_Angstrom', 
               wavelength/(2*np.sin((np.pi/180)*(positions/2))))

    #Calculates the lattice parameter from the spacing
    df2.insert(1, peak_name+'_a_Angstrom', 
               df2[peak_name+'_d_Angstrom']*np.sqrt(hkl[0]**2 + hkl[1]**2 + hkl[2]**2))

    #Calculate and insert the pressure using bulk modulus into the dataframe
    df2.insert(2, peak_name+'_P_GPa', 
               (((a*b*c)-(df2[peak_name+'_a_Angstrom']**3))/(df2[peak_name+'_a_Angstrom'][0]**3))*bulk_mod)
    
    #Check is pydidas or GSAS values for FWHM is provided
    if 'sig0' in df.columns:
        #Invalidates negative sigma values
        df.loc[df["sig"+peak[3:]] < 0, "sig"+peak[3:]] = 0

        #Calculates and insert the FWHM and converts to radians
        df2.insert(3, peak_name+'_FWHM', 
                np.sqrt((8*np.log(2))*df["sig"+peak[3:]]**2)*(np.pi/180))
    else:
        df2.insert(3, peak_name+'_FWHM',
        df["fwhm"+peak[3:]])

    #Calculates and inserts lattice strain
    df2.insert(4, peak_name+'_Lattice_Strain', 
               (df2[peak_name+'_a_Angstrom'] - df2[peak_name+'_a_Angstrom'][0]) / df2[peak_name+'_a_Angstrom'][0])

    return df2

#This finds the key from a dictionary given a unique value
def findkey(search_value, dictionary):
    for key, value in dictionary.items():
        for sublist in value:
            if search_value in sublist:
                return key
            
#This makes a new path if one doesn't exist
def newpath(filepath):
    if not os.path.exists(filepath):
        os.makedirs(filepath)

#Converts an 1D array into an array with shape (x,size(array)/x). This allows us to select an X and Y position and find its corresponding value. output[X][Y]
def conv_xy(xdim,array,output_path):
    ydim = np.floor(array.size/xdim).astype(int)
    output = []
    for i in range(0, ydim):
        output.append([])
        for j in range (0, xdim):

            #This invalidates negative values. Can be toggled by setting 'INVALIDATE HEATMAP NEGATIVES' in user params.
            value = array[i * xdim + j]
            if value < 0 and USER_PARAMS['INVALIDATE HEATMAP NEGATIVES']:
                value = np.nan

            #Invalidates masked values
            if USER_PARAMS["XDI MASK"]:
                if mask[i][j][3] > 0:
                    value = np.nan

            #Caps XDI values that exceed the threshold
            if "fwhm" in output_path and not -1 in USER_PARAMS["FWHM THRESHOLD"]:
                thresholds = USER_PARAMS["FWHM THRESHOLD"]
                value = max(min(value, max(thresholds)), min(thresholds))

            if "intensity" in output_path and not -1 in USER_PARAMS["INTENSITY THRESHOLD"]:
                thresholds = USER_PARAMS["INTENSITY THRESHOLD"]
                value = max(min(value, max(thresholds)), min(thresholds))

            if "pressure" in output_path and not -1 in USER_PARAMS["PRESSURE THRESHOLD"]:
                thresholds = USER_PARAMS["PRESSURE THRESHOLD"]
                value = max(min(value, max(thresholds)), min(thresholds))

            output[i].append(value)
    return output

#Generates a heatmap from an array
def heatmap(data, map_name, output_path, peak_name, filename, dimension):
    fig = plt.figure(figsize=(8,7))
    xdim = np.floor(np.sqrt(data.size)).astype(int) if dimension == -1 else dimension
    Z = conv_xy(xdim, data, output_path)

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

def line_plot(data, variable, output_path, peak_name, filename, autoscale):
    #Create a plot for pressure data vs time
    fig = plt.figure(figsize=(10,6))
    plt.title(peak_name+" {0} vs Position (um)".format(variable))

    #Set the line mask to mask out gasket
    data = data.iloc[min(USER_PARAMS["LINE MASK"]):max(USER_PARAMS["LINE MASK"])+1]
    
    #This sets the lower and upper bounds of the graph.
    if autoscale:
        plt_max = data.max()*1.1 if data.max() > 0 else 0
        plt_min = 0 if data.min() > 0 else data.min()*1.1
        plt.ylim(plt_min, plt_max)

    plt.xlabel("Position (um)")
    plt.ylabel(variable)

    #For the X axis, it generates a list of numbers from 0 to "EXPERIMENT TIME" spaced out by the time resolution
    #The time resolution is calculated by dividing "EXPERIMENT TIME" by how many data points there are
    positions = ()
    plt.plot(np.arange(-data.size//2, data.size//2,1)*USER_PARAMS["LINE STEP SIZE"], data)

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

def decodehdf5(h5f):
    output = pd.DataFrame()
    df = pd.DataFrame(h5f['entry/data/data'])
    col_names = ["pos","int","fwhm"]
    df = df.drop(columns=1)
    df = df.set_axis(col_names, axis='columns')
    return df

if __name__ == "__main__":
    
    #Checks and sees if dynamic and XDI input folders are present
    newpath("inputs/dynamic")
    newpath("inputs/XDI")
    newpath("inputs/line")

    #Determines what peaks to analyze
    peaklist = []
    for material in MATERIALS.keys():
        for peak in MATERIALS[material][0]:
            peaklist.append(peak)

    #Loads the XDI mask if toggled on
    if USER_PARAMS["XDI MASK"]:
        mask = np.array(Image.open("mask.png"))

    for input_type in glb.glob("./inputs/*"):

        #Determines which folder's csv files to look through
        in_path = input_type + "/*.csv"

        #Determines if XDI or Dynamic outputs are toggled based off of what folder inputs are in
        OUTPUTS["DYNAMIC"] = False
        OUTPUTS["XDI"] = False
        OUTPUTS["LINE"] = False

        if input_type == "./inputs\dynamic":
            OUTPUTS["DYNAMIC"] = True

        if input_type == "./inputs\XDI":
            OUTPUTS["XDI"] = True

        if input_type == "./inputs\line":
            OUTPUTS["LINE"] = True

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

            #Read the csv file
            crystaldf = pd.read_csv(file)

            #Calculate the pressure for each peak
            for peak in peaklist:

                #Get the material name, hkl values and set the peak name for graphs
                material = findkey(peak, MATERIALS)
                hkl = SYMMETRY[USER_PARAMS["SYMMETRY"]][int(peak[3:])]
                peak_name = "{3} [{0}{1}{2}]".format(hkl[0],hkl[1],hkl[2], material)

                #Run the calculations with user params
                df2 = calculate_values(crystaldf, 
                peak, 
                peak_name,
                USER_PARAMS['XRAY WAVELENGTH'],
                MATERIALS[material][1][0],
                MATERIALS[material][1][1],
                MATERIALS[material][1][2],
                MATERIALS[material][1][3],
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
                    if OUTPUTS["INTENSITY VS. TIME PLOT"]:
                        dynamic_plot(crystaldf["int"+peak[3:]],
                        "Intensity"
                        ,"{0}\\plots\\{1}\\i_vs_t\\".format(savepath,filename),
                        peak_name,
                        filename,
                        True)

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

                if OUTPUTS["LINE"]:
                    if OUTPUTS["LINE INTENSITY VS. TIME PLOT"]:
                        line_plot(crystaldf["int"+peak[3:]],
                        "Intensity"
                        ,"{0}\\plots\\{1}\\i_vs_t\\".format(savepath,filename),
                        peak_name,
                        filename,
                        True)

                    if OUTPUTS["LINE PRESSURE VS. TIME PLOT"]:
                        line_plot(df2[peak_name+'_P_GPa'],
                        "Pressure (GPa)"
                        ,"{0}\\plots\\{1}\\p_vs_t\\".format(savepath,filename),
                        peak_name,
                        filename,
                        True)

                    if OUTPUTS["LINE FWHM VS. TIME PLOT"]:
                        line_plot(df2[peak_name+'_FWHM'],
                        "FWHM (Rad)"
                        ,"{0}\\plots\\{1}\\fwhm_vs_t\\".format(savepath,filename),
                        peak_name,
                        filename,
                        True)

                    if OUTPUTS["LINE LATTICE PARAMETER VS. TIME PLOT"]:
                        line_plot(df2[peak_name+'_a_Angstrom'],
                        "Lattice Parameter (Å)"
                        ,"{0}\\plots\\{1}\\parameter_vs_t\\".format(savepath,filename),
                        peak_name,
                        filename,
                        False)
                    
                    if OUTPUTS["LINE LATTICE STRAIN VS. TIME PLOT"]:
                        line_plot(df2[peak_name+'_Lattice_Strain'],
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
                    time = np.tile(time, len(peaklist))

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