# %% Imports
import mne
from mne.preprocessing import ICA

import numpy as np
from scipy import interpolate

import os
from scipy.io import loadmat

import matplotlib.pyplot as plt

# %% Loading Raw Data
# Define the directory path where your data files are stored
DATA_PATH = '/Users/brunoandrynascimentocouto/Documents/Data/EEG/BrainAmp Example (TMS)/'

# List all files in the directory and select the first one ending with '.vhdr'
file = [f for f in os.listdir(DATA_PATH) if f.endswith('.vhdr')][0]

# Create an MNE Raw object from the EEG data
# Here, we pass the EOG channels names as the eog parameter, so that MNE knows their names
# We also pass preload=True to load the data into memory right away
raw = mne.io.read_raw(os.path.join(DATA_PATH, file), eog=('VEOG', 'HEOG'), preload=True, verbose=False);

# Rename channels that were mislabeled in the BrainVision Recorder software
raw.rename_channels({'AF7':'AFz', 'AF8':'FCz'})
raw.set_montage('standard_1020')

# Extract events from raw data annotations
events, _ = mne.events_from_annotations(raw)

# Filter events to keep only those with event code 1128
events = events[events[:, 2] == 1128]

# Get the sampling frequency from raw data information
sfreq = raw.info['sfreq']

# Define the interpolation window in terms of samples
interp_window = (int(-0.002*sfreq), int(0.005*sfreq))

# Define a function to remove TMS pulse artifact and interpolate data
def remove_tms_pulse_artifact(y, onsets, interp_window):
    """
    Remove data from y between the indices (onset+x1) and (onset+x2) for each onset, 
    and interpolate across this gap.
    This function assumes that the y values correspond to equally spaced x values.
    """
    # Extract the start and end points of the interpolation window
    x1, x2 = interp_window

    # Create a copy of y to avoid modifying the original data
    y_new = y.copy()
    
    # Create corresponding x-values
    x = np.arange(len(y))

    # Loop over each onset
    for onset in onsets:
        # Check if the window is valid
        if not 0 <= onset+x1 < onset+x2 < len(y):
            raise ValueError(f"Invalid window for onset {onset}")

        # Make sure we have enough points for cubic interpolation
        if onset+x1 < 2 or onset+x2 > len(y) - 3:
            raise ValueError(f"Not enough points for cubic interpolation at onset {onset}")

        # Include two points on either side of the gap
        x_for_interpolation = np.concatenate((x[onset+x1-2:onset+x1], x[onset+x2+1:onset+x2+3]))
        y_for_interpolation = np.concatenate((y_new[onset+x1-2:onset+x1], y_new[onset+x2+1:onset+x2+3]))

        # Use cubic interpolation to create new y-values for the window
        f = interpolate.interp1d(x_for_interpolation, y_for_interpolation, kind='cubic')
        y_new[onset+x1:onset+x2+1] = f(x[onset+x1:onset+x2+1])

    return y_new

# Apply the function to remove TMS pulse artifacts and interpolate data
raw.pick('eeg').apply_function(remove_tms_pulse_artifact, onsets=events[:, 0], interp_window=interp_window, verbose=False);

# Apply a 4th order Butterworth filter, with high-pass at 0.1 Hz and low-pass at 200 Hz
raw.filter(l_freq=1, h_freq=None, method='iir')

# Segment the continuous data into epochs, from -500 ms to 500 ms around the event onsets
epochs = mne.Epochs(raw, events, tmin=-0.8, tmax=0.8, preload=True, baseline=None)

epochs.set_eeg_reference('average', projection=True)
epochs.apply_proj()

# Aditional
epochs.filter(h_freq=45, method='iir')
epochs.resample(1000)
epochs = epochs.crop(-0.6, 0.6)

#  Baseline Correction
epochs.apply_baseline((-0.6, -0.01), verbose=True)

# Compute the average of all epochs to create an evoked response
evoked = epochs.pick('eeg').crop(-.2,.5).average()
evoked.plot_joint(times=[.007],
                  topomap_args=dict(cmap='jet',
                                    image_interp='linear'));


fig, ax = plt.subplots(1, 1, figsize=(10, 10))
evoked.plot_topomap(times=[.007], ch_type='eeg',
                    cmap='jet', contours=0,
                    colorbar=False,
                    show_names=True, axes=ax);