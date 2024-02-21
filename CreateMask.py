from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import pandas as pd
import glob as glb

def conv_xy(xdim,array):
    ydim = np.floor(array.size/xdim).astype(int)
    output = []
    for i in range(0, ydim):
        output.append([])
        for j in range (0, xdim):

            value = array[i * xdim + j]
            if value < 0:
                value = np.nan
            
            output[i].append(value)
    return output

if __name__ == "__main__":
    df = pd.read_csv(glb.glob("./inputs/XDI/*.csv")[0])

    #Choose which peak to create mask with
    peak = 'sig1'
    df2 = df[peak]

    #Mask Cu Pressure Values
    if False:
        system = [["pos3"],[3.6150, 3.6150, 3.6150, 140.0], [1,1,1]]

        wavelength = 0.4133
        dspacing = wavelength/(2*np.sin((np.pi/180)*(df[system[0][0]]/2)))
        latticeparam = dspacing*np.sqrt(system[2][0]**2 + system[2][1]**2 + system[2][2]**2)
        pressure = (((system[1][0]*system[1][1]*system[1][2])-(latticeparam**3))/(latticeparam**3))*system[1][3]
        
        Z = conv_xy(np.floor(np.sqrt(pressure.size)).astype(int), pressure)
    else:
        Z = conv_xy(np.floor(np.sqrt(df2.size)).astype(int), df2)

    Z = np.nan_to_num(Z, nan=0.0)
    X = np.arange(np.array(Z).shape[1])
    Y = np.arange(np.array(Z).shape[0])
    img = Image.fromarray(np.uint8(Z)).convert('RGB')

    a_channel = Image.new('L', img.size, 255)
    img.putalpha(a_channel)
    img.save("mask.png")