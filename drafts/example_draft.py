# %% TECAPy Draft

# 
# TECAPy (TMS-EEG Cleaning & Analysis Pipeline) is python pipeline for preprocessing TMS-EEG evoked potentials (TEPs).
# It is inspired in the TMS-EEG preprocessing software SiSyPhus (SSP).
#
# Author: Couto, B.A.N. Member of the Neuroengineering and Computation Lab at the Federal University of São Paulo.
#

# Simple example based on the methodologies described on the papers:

# Hernandez-Pavon et al. TMS combined with EEG: Recommendations and open issues for data collection and analysis
# - Section: 6.2 TMS–EEG pipelines for analysis
# DOI: https://doi.org/10.1016/j.brs.2023.02.009

# Rogasch et al. Designing and comparing cleaning pipelines for TMS-EEG data: A theoretical overview and practical example
# DOI: https://doi.org/10.1016/j.jneumeth.2022.109494
# Data: https://bridges.monash.edu/articles/dataset/TESA_cleaningPipelines/18805994/4
# Git: https://github.com/nigelrogasch/TMS-EEG-cleaning-pipelines
# Target: Left Superior Frontal Gyrus
# 120 pulses with interval jittered between 4-6 seconds

# %% Imports
import numpy as np
import mne
from scipy.io import loadmat
from scipy import interpolate
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')

# Parameters:
PATH = '/Users/brunoandrynascimentocouto/Documents/Data/EEG/TESA Cleaning Pipelines (TMS)/'

# %% Loading Raw data
def get_rejections_information():
    rejections = loadmat(os.path.join(PATH, 'Rejections.mat'))
    rejection_ids = [rejections['ID'][i][0][0] for i in range(len(rejections['ID']))]
    rejection_dlpfc = [rejections['DLPFC_badTrails'][0][i].reshape(-1) for i in range(len(rejections['DLPFC_badTrails'][0]))]
    rejection_dlpfc = [list(map(int, rejection_dlpfc[i])) for i in range(len(rejection_dlpfc))]
    rejection_fef = [rejections['FEF_badTrials'][0][i].reshape(-1) for i in range(len(rejections['FEF_badTrials'][0]))]
    rejection_fef = [list(map(int, rejection_fef[i])) for i in range(len(rejection_fef))]
    badchannels = []
    for i in range(len(rejections['badChannels'][0])):
        bads = rejections['badChannels'][0][i].reshape(-1)
        bads = list(map(list, bads))
        if bads == [[]]:
            bads = []
        else:
            bads = [bads[i][0] for i in range(len(bads)) if bads[i] != []]
        badchannels.append(bads)
    rejections = {
        'ID': rejection_ids,
        'badTrials': rejection_fef,
        'badChannels': badchannels
    }
    return rejections

files = os.listdir(PATH)
files = [file for file in files if file.endswith('FEF_Decaytest_withTMSPulse.set')]

#raws = [mne.io.read_epochs_eeglab(os.path.join(PATH, file)) for file in files]
# %% Step-by-Step Guide
# TMS-EEG Pipelines for Analysis
# 1. Epoching data around the TMS pulse (-500, 500 ms)
# 1.1. Problem: High-pass before or after epoching and removing pulse artifact
epoch = mne.io.read_epochs_eeglab(os.path.join(PATH, files[0]))
raw_evoked = epoch.average().copy().get_data()

# 2. Removing bad channels/trials
epoch.plot(n_epochs=3, precompute=True, use_opengl=True, theme='dark', decim=40)

# 3. Removing and interpolating the TMS pulse artifact
# 3.1 Common practice: 1-2ms before and up to 5-10ms after the TMS pulse (Use cubic interpolation)
def remove_and_interpolate(y, x1, x2):
    """
    Remove data from y between the indices x1 and x2, and interpolate across this gap using cubic interpolation.

    Args:
        y (array): The array from which to remove and interpolate data.
        x1 (int): The index of the first point to remove.
        x2 (int): The index of the last point to remove.

    Returns:
        y_new (array): The modified array with the removed section interpolated.
    """
    # Check if x1 and x2 are valid indices for y
    if not 0 <= x1 < x2 < len(y):
        raise ValueError("x1 and x2 must be valid indices for y")

    # Split y into three parts: before, during, and after the removed section
    y_before = y[:x1]
    y_during = y[x1:x2+1]
    y_after = y[x2+1:]

    # Create corresponding x-values
    x = np.arange(len(y))
    x_before = x[:x1]
    x_after = x[x2+1:]

    # Use cubic interpolation to create new y-values for the removed section
    f = interpolate.interp1d(np.hstack((x_before, x_after)), np.hstack((y_before, y_after)), kind='cubic')
    y_during_new = f(x[x1:x2+1])

    # Concatenate the before, new during, and after sections to form the modified array
    y_new = np.hstack((y_before, y_during_new, y_after))

    return y_new

idx1, idx2 = epoch.time_as_index([-.002, .01])
epoch.apply_function(remove_and_interpolate, x1=idx1, x2=idx2, verbose=True)

# 4. Re-referencing the data (mean reference/average reference)
# 4.1 Problem: Don't forget to interpolate the channels that were removed
epoch.interpolate_bads(reset_bads=True, verbose=True)
epoch.set_eeg_reference('average', projection=True, verbose=True)

# 5. Baseline correction (-500, -010 ms)
epoch.apply_baseline((-0.5, -0.01), verbose=True)

epoch.average().plot_joint(topomap_args=dict(cmap='jet'))

# 6. Dealing with large amplitude artifacts and or 7. auditory/somatosensory evoked responses
# 6/7.1 We will use a ICA based blind source separation (BSS) method to remove these

# 8. Temporal Filtering
# epoch.filter(1, 40, fir_design='iir', verbose=True)

# 9. Interpolate removed channels

# 10. Averaging

# 11. Downsampling (500 Hz)

# %% Comparisson
new_evoked = epoch.average().copy().get_data()

fig, axs = plt.subplots(2, 1, sharex=True)
axs[0].plot(raw_evoked.T)
axs[0].set_title('Raw')
axs[1].plot(new_evoked.T)
axs[1].set_title('Cleaned')
plt.show()

# %%
# Raw
# Remove TMS Artifact on Raw

# Question
# Run ICA and or Trial/Channel Rejection?
# - ICA
# - Remove ICA components

# Protocol 2:
# - Preprocess
# -- Detrend (Perguntar sobre o Detrend por já ter código de matlab)
# -- Highpass 0.1, Lowpass 200 Hz (filt-filt butter 3rd order)
# -- Downsample 1000 Hz
# -- Epoch (-500, 500) ms
# -- Avg Reference
# -- Baseline (-500, -10) ms
# -- Interpolate Bad Channels
