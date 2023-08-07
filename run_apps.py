# %% Imports
import pickle
from ica_app import ICApp

with open('ica.pickle', 'rb') as f:
    ica = pickle.load(f)

with open('epochs.pickle', 'rb') as f:
    epochs = pickle.load(f)

# %% Run ICApp
ica.exclude = [0, 5]
new_ica = ICApp(ica, epochs)

# %% App
new_ica = ICApp(new_ica, epochs)

# %% Check Components

# Modifications:
# Add metadata to dropped_components
# Add ratio
# Fix number issue
# Fix
# --> 367 self.ica.exclude = [int(bad_item) for bad_item in bad_items]
# ValueError: invalid literal for int() with base 10: 'ICA003'
