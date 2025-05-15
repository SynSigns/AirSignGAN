import numpy as np
import glob
import pandas
import os
from tqdm import tqdm
import shutil

# files = glob.glob('/home/alpha/Workbenches/rohan/SynSignatures/SinGanLSTM/SlitTrainData/CombinedNoAug/*/*',recursive=True)
# dest = '/home/alpha/Workbenches/rohan/SynSignatures/SinGanLSTM/SlitTrainData/CombinedAug'

files = glob.glob('/home/alpha/Workbenches/rohan/SynSignatures/SinGanLSTM/SlitAbalationData/attentionlstm/CombinedNoAug/*/*',recursive=True)
dest = '/home/alpha/Workbenches/rohan/SynSignatures/SinGanLSTM/SlitAbalationData/attentionlstm/CombinedAug'


if os.path.exists(dest):
    shutil.rmtree(dest)
os.makedirs(dest, exist_ok=True)

x_col = [0,3]
y_col = [1,4]
z_col = [2,5]

for file in tqdm(files):
    arr = pandas.read_csv(file, header=None,sep=",").to_numpy()
    foldername, filename = os.path.split(file)
    foldername = os.path.basename(foldername)
    os.makedirs(os.path.join(dest,foldername),exist_ok=True)
    for i in range(5):
        angle = (i-2)*np.pi/36
        sin = np.sin(angle)
        cos = np.cos(angle)
        for j in range(3):
            factor = 1. + (j-1)*0.05
            for k in range(2):
                new_file = os.path.join(dest,foldername,os.path.splitext(filename)[0] + f'-aug-{i}-{2*j+k}.txt')
                new_arr = arr.copy()
                for xi,yi,zi in zip(x_col,y_col,z_col):
                    new_arr[:,xi] = arr[:,xi]*cos - arr[:,yi]*sin
                    new_arr[:,yi] = arr[:,xi]*sin + arr[:,yi]*cos
                    if k == 0:
                        new_arr[:,xi] = arr[:,xi] * factor
                        new_arr[:,zi] = arr[:,zi] * factor
                    else:
                        new_arr[:,yi] = arr[:,yi] * factor
                        new_arr[:,zi] = arr[:,zi] * factor
                np.savetxt(new_file,new_arr,delimiter=',')

print("Done")
