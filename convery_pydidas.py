import pandas as pd
import glob as glb
import h5py

output = pd.DataFrame()

for index, file in enumerate(glb.glob("./input/*.h5")):
    h5f = h5py.File(file, "r")['entry/data/data']
    df = pd.DataFrame(h5f)

    col_names = ["pos","int","fwhm"]
    col_names = [x + str(index) for x in col_names]
    df = df.drop(columns=1)
    df = df.set_axis(col_names, axis='columns')

    output = pd.concat([output, df], axis='columns')
            
output.to_csv("./output/merged_data.csv")