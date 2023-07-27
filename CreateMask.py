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

    peak = df['int2']

    Z = conv_xy(np.floor(np.sqrt(peak.size)).astype(int), peak)
    Z = np.nan_to_num(Z, nan=0.0)
    X = np.arange(np.array(Z).shape[1])
    Y = np.arange(np.array(Z).shape[0])
    img = Image.fromarray(np.uint8(Z)).convert('RGB')

    filter = False
    if filter:
        img_filter = ImageEnhance.Brightness(img)
        img = img_filter.enhance(100)
        img_filter = ImageEnhance.Contrast(img)
        img = img_filter.enhance(100)
        img = ImageOps.invert(img)
    a_channel = Image.new('L', img.size, 255)
    img.putalpha(a_channel)
    img.save("mask.png")