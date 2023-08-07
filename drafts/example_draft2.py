# %% Imports
import mne
from mne.preprocessing import ICA

import numpy as np
from scipy import interpolate

import matplotlib.pyplot as plt
import os

from scipy.io import loadmat

# %% Loading Raw Data
# File Path:
PATH = '/Users/brunoandrynascimentocouto/Documents/Data/EEG/BrainAmp Example (TMS)/'
file = [file for file in os.listdir(PATH) if file.endswith('.vhdr')][0]

# Chlocs Information (Channel Names)
chlocs = loadmat(os.path.join(PATH, 'chlocs_brainamp62milano.mat'))['chlocs'][0]
chlocs = [ch[0][0] for ch in chlocs]

# %% Creating Raw MNE
raw = mne.io.read_raw(os.path.join(PATH, file), eog=('VEOG', 'HEOG'), preload=True)
raw.reorder_channels(chlocs+['VEOG', 'HEOG'])

# Inspect raw
raw.plot()

# %% Dealing with events
events, _ = mne.events_from_annotations(raw)
events = events[events[:, 2] == 1128]
sfreq = raw.info['sfreq']
interp_window = (int(-0.002*sfreq), int(0.01*sfreq))

# %% Removing TMS Artifact on Raw
def remove_tms_pulse_artifact(y, onsets, interp_window):
    """
    Remove data from y between the indices (onset+x1) and (onset+x2) for each onset, 
    and interpolate across this gap.
    This function assumes that the y values correspond to equally spaced x values.
    """

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
raw.apply_function(remove_tms_pulse_artifact, onsets=events[:, 0], interp_window=interp_window, verbose=True)

# %% Question:
# Run ICA and or Trial/Channel Rejection?

# %% ICA
filt_raw = raw.copy().filter(l_freq=1.0, h_freq=None, method='iir')
ica = ICA(n_components=15, max_iter="auto")
ica.fit(filt_raw, decim=3)
ica.plot_components(inst=raw, cmap='jet')
# Remove ICA components
ica.exclude = [0, 2, 4, 10, 11]

# Reconstruct Raw
reconst_raw = raw.copy()
ica.apply(reconst_raw)
reconst_raw.plot()
raw = reconst_raw

# %% Protocol 2:
# Basic Preprocessing
# -- Detrend (Do you already have a matlab code?)
# -- Highpass 0.1, Lowpass 200 Hz (filt-filt butter 3rd order)
raw.filter(l_freq=0.1, h_freq=200, method='iir') # 4th order butterworth

# -- Downsample 1000 Hz
raw.resample(1000, npad='auto')
events, _ = mne.events_from_annotations(raw)
events = events[events[:, 2] == 1128]

# -- Epoch (-500, 500) ms
epoch = mne.Epochs(raw, events, tmin=-0.5, tmax=0.5, preload=True)

# -- Inspecting Trials
epoch.plot(n_epochs=3, precompute=True, use_opengl=True, theme='dark', block=True)
# Block: Waits the plot to close before continuing

# -- Avg Reference
epoch.set_eeg_reference('average', projection=True)
epoch.apply_proj()

# -- Baseline (-500, -10) ms
epoch.apply_baseline((-0.5, -0.01), verbose=True)

# -- Interpolate Bad Channels
epoch.interpolate_bads(reset_bads=True, verbose=True)

# %% Evoked
evoked = epoch.average()
evoked.plot_joint(topomap_args=dict(cmap='jet'));

# %% Sketches
# # Fake TMS Pulse
# t = np.arange(0, 5, 1/sfreq)
# y = np.sin(2*np.pi*10*t) # + np.random.normal(0, 0.1, len(t))
# onsets = []
# for i in range(10):
#     y[(800*i)+10+interp_window[0]:(800*i)+10+interp_window[1]+1] = 5
#     onsets.append((800*i)+10)
# plt.plot(y)
# y_new = remove_tms_pulse_artifact(y, onsets, interp_window)
# plt.plot(y_new)