#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

dataPath = '/Users/galgranot/Documents/galag/school/raman/'

col1 = "Time - Plot 0"
col2 = "Amplitude - Plot 0"

def main():
    wavelengths = {}
    files = [f for f in os.listdir(dataPath) if f.endswith(".txt") and "_" in f]
    for file in files:
        if file.endswith(".png"):
            continue
        
        parts = file.split("_")
        wavelength = parts[0]
        file_path = os.path.join(dataPath, file)
        df = pd.read_csv(file_path, sep="\t")
        
        if "_0.txt" in file:
            if wavelength not in wavelengths:
                wavelengths[wavelength] = {'signal': None, 'noise': None}
            wavelengths[wavelength]['signal'] = df
        
        elif "_2.txt" in file:
            if wavelength not in wavelengths:
                wavelengths[wavelength] = {'signal': None, 'noise': None}
            wavelengths[wavelength]['noise'] = df
            
    for wavelength, data in wavelengths.items():
        if data['signal'] is not None and data['noise'] is not None:
            plt.figure(figsize=(15, 10))
            print(f"Processing wavelength {wavelength}")
            
            dfSignal = data['signal']
            dfNoise = data['noise']
            
            np0x = dfSignal[col1].to_numpy()
            np0y = dfSignal[col2].to_numpy()
            npxNoise = dfNoise[col1].to_numpy()
            npyNoise = dfNoise[col2].to_numpy()

            plt.subplot(1, 3, 1)
            plt.scatter(np0x, np0y, color='blue', s=1)
            plt.title(f"{wavelength} signal")
            plt.grid(True)
            
            plt.subplot(1, 3, 2)
            plt.scatter(npxNoise, npyNoise, color='blue', s=1)
            plt.title(f"{wavelength} noise")
            plt.grid(True)
            
            plt.subplot(1, 3, 3)
            plt.scatter(np0x, np0y - npyNoise, color='blue', s=1)
            plt.title(f"{wavelength} filtered signal")
            plt.grid()
            plt.savefig(f"{wavelength}.png")

if __name__ == "__main__":
    main()
