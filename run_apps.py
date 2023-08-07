# %% Imports
import pickle
from ica_app import ICApp

with open('ica.pickle', 'rb') as f:
    ica = pickle.load(f)

with open('ica45.pickle', 'rb') as f:
    ica45 = pickle.load(f)

with open('epochs.pickle', 'rb') as f:
    epochs = pickle.load(f)

# %% Run ICApp
ica.exclude = [0, 5]
new_ica = ICApp(ica, epochs)

# %% App
new_ica = ICApp(ica45, epochs)

# %% Check Components

# Modifications:
# Add metadata to dropped_components?
# Order by explained variance?
