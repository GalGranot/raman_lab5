import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd # handling data structures (loaded from files)
import scipy
from scipy.signal import find_peaks as find_peaks # Find peaks inside a signal based on peak properties.
import scipy.stats as stats
from scipy.stats import t
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
import math

def linear_func(x, a, b):
    return a * x + b

# Model for gaussian pdf, with amp, mu and sigma parameters
def gaussian_pf(x, amp, mu, sigma, b):
    return amp * np.exp(-(x - mu)**2 / (2 * sigma**2)) + b

def double_gaussian_pf(x, amp1, mu1, sigma1, amp2, mu2, sigma2, b):
    g1 = amp1 * np.exp(-(x - mu1)**2 / (2 * sigma1**2))
    g2 = amp2 * np.exp(-(x - mu2)**2 / (2 * sigma2**2))
    return g1 + g2 + b

def lorentzian_func(x, amp, x_0, gamma, b):
    return amp*(gamma/np.pi) /( ((x - x_0)**2 + (gamma)**2) ) + b

def abs_lorentzian_func(x, amp, x_0, gamma, b):
    return np.abs(amp)*(np.abs(gamma)/np.pi) /( ((x - x_0)**2 + (gamma)**2) ) + b

def double_lorentzian_func(x, amp1, x_01, gamma1, amp2, x_02, gamma2, b):
    l1 = amp1 * (gamma1 / np.pi) / (((x - x_01) ** 2 + (gamma1) ** 2))
    l2 = amp2 * (gamma2 / np.pi) / (((x - x_02) ** 2 + (gamma2) ** 2))
    return l1 + l2 + b


def smooth_vector_with_padding(vector, k):
    # Create a uniform kernel of size k
    kernel = np.ones(k) / k
    # Apply convolution
    smoothed = np.convolve(vector, kernel, mode='valid')  # 'valid' avoids padding artifacts

    # Add k//2 zeros to the start and end of the smoothed vector
    padding = k // 2
    smoothed_with_padding = np.pad(smoothed, (padding, padding), mode='constant', constant_values=0)
    return smoothed_with_padding

def calibration(folder_path, a, mu, s, b):
    # Get all text files in the folder, sorted to ensure consistent pairing
    files = sorted([f for f in os.listdir(folder_path) if f.endswith(".txt")])

    mu_vec = np.zeros(len(files) - 1)
    WN_vec = np.zeros(len(files) - 1)

    for i in range(0, len(files) - 1):
        file = files[i]
        path = os.path.join(folder_path, file)

        try:
            # Load columns from the first file
            wn1, counts1 = np.loadtxt(path, skiprows=1, unpack=True)
            # x = wn1[mu[i] - 100 : mu[i] + 100]
            # y = counts1[mu[i] - 100 : mu[i] + 100]
            x = wn1
            y = counts1
            amp_init = a[0]
            mu_init = mu[i]
            sig_init = s[0]
            b_init = b[0]
            Gaussopt, Gausscov = (curve_fit
                                  (gaussian_pf, x, y, p0=np.array([amp_init, mu_init, sig_init, b_init])))
            y_gauss_fit = gaussian_pf(x, *Gaussopt)

            mu_vec[i] = Gaussopt[1]
            WN_vec[i] = int(file.replace('.txt', ''))

            plt.figure(figsize=(15, 5))
            plt.plot(wn1, counts1, label="Measured Counts", color="blue")
            plt.plot(x, y_gauss_fit, label="Gaussian Fit", color="red")
            plt.title(f"Calibration Input Wavenumber = {file.replace('.txt', '')} Measurement")
            plt.xlabel("Monochromator CCD Pixels")
            plt.ylabel("Counts")
            plt.grid(True)

            # Show the plots
            plt.tight_layout()
            plt.show()


        except Exception as e:
            print(f"Error processing file {file}: {e}")

    a_lin = -70
    b_lin = 20000
    Linopt, Lincov = (curve_fit
                          (linear_func, WN_vec, mu_vec, p0=np.array([a_lin, b_lin])))
    y_lin_fit = linear_func(WN_vec, *Linopt)

    plt.figure(figsize=(15, 5))
    plt.plot(WN_vec, mu_vec, label="Calibration Measurements", color="blue")
    plt.plot(WN_vec, y_lin_fit, label="Linear Calibration Fit", color="red")
    plt.title("Calibration Measurements Fitting")
    plt.xlabel("Input Wavenumber to Monochromator [1/cm]")
    plt.ylabel("Counts")
    plt.grid(True)

    plt.figure(figsize=(15, 5))
    plt.plot(WN_vec, mu_vec - y_lin_fit, label="Calibration Measurements", color="blue")
    plt.title("Difference between Calibration Measurements and Fit")
    plt.xlabel("Input Wavenumber to Monochromator [1/cm]")
    plt.ylabel("Counts Difference between Fit and Measurement")
    plt.grid(True)

    # Show the plots
    plt.tight_layout()
    plt.show()

    print(Linopt)
    print(mu_vec)
    print(WN_vec)

    a_pred = Linopt[0]
    b_pred = Linopt[1]

    # Fitted Model: pixel = a_pred * WN + b_pred
    # -> therefore del_pixel = a_pred * del_WN  --> del_WN = del_pixel/a_pred
    # Another way to calculate del_pixel is the error in the pixels we got, from the calibration
    # measurements - we know that at the HeNe WN we are supposed to be at pixel 512 (exactly at the middle)
    # but since we are not, the pixel we got from the model, that meets with the HeNe wavelength is the true "center":
    # center pixel is: c_px = a_pred * WN_HeNe + b_pred
    # and we receive:
    # WN_true = WN_input + (del_WN/del_pixel) * (c_px - pixel) = WN_input + (1/a_pred) * (c_px - pixel)

    # With this model, we input the WN we gave the monochromator, and the pixel we are looking at, and we receive
    # its corresponding wavenumber.

    # to get the frequency: f = 3 * (10^10) * (WN [1/cm])  [Hz]

    WN_HeNe = 15822.78 # 1/cm
    c_px = a_pred * WN_HeNe + b_pred

    return mu_vec, WN_vec, a_pred, b_pred, c_px


def process_and_plot_file_pairs(folder_path, a, mu, s, idx, b):
    # Get all text files in the folder, sorted to ensure consistent pairing
    files = sorted([f for f in os.listdir(folder_path) if f.endswith(".txt")])

    # Ensure there are pairs of files to process
    if len(files) % 2 != 0:
        print("Warning: Odd number of files. The last file will be skipped.")
        files = files[:-1]

    WN_input_vec = np.zeros(len(files))
    y_vec = np.array([])
    y_gauss_vec = np.array([])
    y_lorr_vec = np.array([])

    # Process files in pairs
    for i in range(0, len(files), 2):
        file1, file2 = files[i], files[i + 1]
        path1, path2 = os.path.join(folder_path, file1), os.path.join(folder_path, file2)

        WN_input = int(file1.replace('_0.txt', ''))

        WN_input_vec[i] = WN_input

        try:
            # Load columns from the first file
            wn1, counts1 = np.loadtxt(path1, skiprows=1, unpack=True)

            # Load columns from the second file
            wn2, counts2 = np.loadtxt(path2, skiprows=1, unpack=True)

            # Subtract the minimum value from counts1 and counts2
            counts1 = counts1 - np.min(counts1)
            counts2 = counts2 - np.min(counts2)

            if i == 0:
                counts2 = counts2*4

            if i == 6:
                counts2 = counts2*1.3

            if i == 8:
                counts2 = counts2*2.5

            if i == 10:
                counts2 = counts2*3.3

            if i == 12:
                counts2 = counts2*1.2

            # Compute counts_dif
            counts_dif = counts1 - counts2

            ind = int(i/2)
            x = wn1[idx[ind, 0]: idx[ind, 1]]
            y = counts_dif[idx[ind, 0]: idx[ind, 1]]
            y = smooth_vector_with_padding(y, k=5)

            if i == 2 or i == 12:
                a1 = a[ind]
                mu1 = mu[ind]
                sig1 = s[ind]
                if i == 2:
                    a2 = a[ind]/1.3
                    mu2 = mu[ind] + 200
                else:
                    # y = smooth_vector_with_padding(y, k=15)
                    a2 = a[ind]
                    mu2 = mu[ind] + 90
                sig2 = s[ind]
                b_init = b[ind]
                Gaussopt, Gausscov = (curve_fit
                                  (double_gaussian_pf, x, y, p0=np.array([a1, mu1, sig1, a2, mu2, sig2, b_init])))
                Lorropt, Lorrcov = (curve_fit
                                  (double_lorentzian_func, x, y, p0=np.array([a1, mu1, sig1, a2, mu2, sig2, b_init])))
                y_gauss_fit = double_gaussian_pf(x, *Gaussopt)
                y_lorr_fit = double_lorentzian_func(x, *Lorropt)
                print(f"The fit parameters for WN: {file1.replace('_0.txt', '')} are:\n")
                print(f"Gaussian:  {Gaussopt}\n")
                print(f"Lorentzian:  {Lorropt}\n")

            else:
                amp_init = a[ind]
                mu_init = mu[ind]
                sig_init = s[ind]
                b_init = b[ind]
                Gaussopt, Gausscov = (curve_fit
                                  (gaussian_pf, x, y, p0=np.array([amp_init, mu_init, sig_init, b_init])))
                y_gauss_fit = gaussian_pf(x, *Gaussopt)
                if i == 6 or i == 10:
                    Lorropt, Lorrcov = (curve_fit
                                  (abs_lorentzian_func, x, y, p0=np.array([amp_init, mu_init, sig_init, b_init])))
                    y_lorr_fit = abs_lorentzian_func(x, *Lorropt)
                else:
                    Lorropt, Lorrcov = (curve_fit
                                  (lorentzian_func, x, y, p0=np.array([amp_init, mu_init, sig_init, b_init])))
                    y_lorr_fit = lorentzian_func(x, *Lorropt)
                print(f"The fit parameters for WN: {file1.replace('_0.txt', '')} are:\n")
                print(f"Gaussian:  {Gaussopt}\n")
                print(f"Lorentzian:  {Lorropt}\n")


            # Plotting
            plt.figure(figsize=(15, 5))  # Create a figure with space for subplots

            # Subplot 1: (wn1, counts1)
            plt.subplot(2, 3, 1)
            plt.plot(wn1, counts1, label="Counts", color="blue")
            plt.title(f"Raw Input Wavenumber = {file1.replace('_0.txt', '')} Measurement")
            plt.xlabel("Monochromator CCD Pixels")
            plt.ylabel("Counts")
            plt.grid(True)

            # Subplot 2: (wn2, counts2)
            plt.subplot(2, 3, 2)
            plt.plot(wn2, counts2, label="Counts2", color="green")
            plt.title(f"Background noise in Input Wavenumber = {file2.replace('_2.txt', '')} Measurement")
            plt.xlabel("Monochromator CCD Pixels")
            plt.ylabel("Counts")
            plt.grid(True)

            # Subplot 3: (wn1, counts_dif)
            plt.subplot(2, 3, 3)
            plt.plot(wn1, counts_dif, label="Counts Difference", color="red")
            plt.title(f"Noise Redacted Wavenumber = {file1.replace('_0.txt', '')} Measurement")
            plt.xlabel("Monochromator CCD Pixels")
            plt.ylabel("Counts Difference")
            plt.grid(True)

            # Subplot 4: (x, y)
            plt.subplot(2, 3, 4)
            plt.plot(x, y)
            plt.plot(x, y_gauss_fit, label="Gaussian Fit", color="red")
            plt.title(f"Noise Redacted Wavenumber = {file1.replace('_0.txt', '')} Peak Gaussian Fit")
            plt.xlabel("Monochromator CCD Pixels")
            plt.ylabel("Counts Difference")
            plt.grid(True)

            # Subplot 5: (x, y)
            plt.subplot(2, 3, 5)
            plt.plot(x, y)
            plt.plot(x, y_lorr_fit, label="Lorentzian Fit", color="red")
            plt.title(f"Noise Redacted Wavenumber = {file1.replace('_0.txt', '')} Peak Lorentzian Fit")
            plt.xlabel("Monochromator CCD Pixels")
            plt.ylabel("Counts Difference")
            plt.grid(True)

            # Show the plots
            plt.tight_layout()
            plt.show()

            y_vec = np.append(y_vec, y)
            y_gauss_vec = np.append(y_gauss_vec, y_gauss_fit - Gaussopt[-1])
            y_lorr_vec = np.append(y_lorr_vec, y_lorr_fit - Lorropt[-1])



        except Exception as e:
            print(f"Error processing files {file1} and {file2}: {e}")

    return WN_input_vec, y_vec, y_gauss_vec, y_lorr_vec



# Calibration Stuff:

a_cal = np.array([1800])
mu_cal = np.array([900, 700, 500, 450, 380, 300, 220])
s_cal = np.array([5])
b_cal = np.array([300])

folder_path_cal = r"C:\Users\David\Python_Projects\Physics_Lab_5\Calibration"
mu_cal_vec, WN_cal_vec, a_pred, b_pred, c_px = calibration(folder_path_cal, a_cal, mu_cal, s_cal, b_cal)


# Fit Stuff:

px = np.array([[80, 220], [300, 800], [400, 900], [400, 800], [300, 700], [300, 700], [300, 700]])

num_files = px.shape[0]

a = np.array( [180, 200, 500, 200, 200, 180, 80])
mu = np.array([160, 500, 610, 600, 500, 550, 520])
s = np.array( [10 , 30 , 50 , 30 , 50 , 50 , 5  ])
b = np.array([-10, 20, 0, 40, -10, 60, 50])

# Folder path
folder_path = r"C:\Users\David\Python_Projects\Physics_Lab_5\Measurements"
WN_input_vec, peak_vec, peak_gauss_vec, peak_lorr_vec = process_and_plot_file_pairs(folder_path, a, mu, s, px, b)

WN_true_mat = np.zeros((num_files, 2))

for i in range(num_files):
    WN_true_mat[i, 0] = WN_input_vec[i * 2] + (1 / a_pred) * (c_px - px[i, 0])
    WN_true_mat[i, 1] = WN_input_vec[i * 2] + (1 / a_pred) * (c_px - px[i, 1])

dWN = np.abs(1/a_pred)
WN_true_vec = np.arange(WN_true_mat[0][0], WN_true_mat[-1][-1] + dWN, dWN)

f_mat = 3 * (10**10) * WN_true_mat
f_vec = 3 * (10**10) * WN_true_vec

# np.where(f_vec == value_to_find)[0]

peak_vec_full = np.zeros(len(f_vec))
peak_gauss_vec_full = np.zeros(len(f_vec))
peak_lorr_vec_full = np.zeros(len(f_vec))

ind_peak_vec = 0
for i in range(num_files):
    ind_0 = np.argmin(np.abs(f_vec - f_mat[i, 0]))
    num_peak_samples = px[i, 1] - px[i, 0]

    ind_final = ind_0 + num_peak_samples

    peak_vec_full[ind_0: ind_final] = peak_vec[ind_peak_vec: ind_peak_vec + num_peak_samples]
    peak_gauss_vec_full[ind_0: ind_final] = peak_gauss_vec[ind_peak_vec: ind_peak_vec + num_peak_samples]
    peak_lorr_vec_full[ind_0: ind_final] = peak_lorr_vec[ind_peak_vec: ind_peak_vec + num_peak_samples]

    ind_peak_vec = ind_peak_vec + num_peak_samples


plt.figure(figsize=(15, 5))

plt.plot(f_vec*1e-12, peak_vec_full, label="Calibration Measurements", color="blue")
plt.title("All Peaks")
plt.xlabel("f [THz]")
plt.ylabel("Counts")
plt.grid(True)


plt.figure(figsize=(15, 5))
plt.plot(f_vec*1e-12, peak_gauss_vec_full, label="Calibration Measurements", color="blue")
plt.title("All Gaussian Fit Peaks")
plt.xlabel("f [THz]")
plt.ylabel("Counts")
plt.grid(True)


plt.figure(figsize=(15, 5))
plt.plot(f_vec*1e-12, peak_lorr_vec_full, label="Calibration Measurements", color="blue")
plt.title("All Lorentzian Fit Peaks")
plt.xlabel("f [THz]")
plt.ylabel("Counts")
plt.grid(True)

# Show the plots
plt.tight_layout()
plt.show()
