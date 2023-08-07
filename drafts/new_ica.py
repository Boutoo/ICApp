# Concatena os trials
# canal, tempo, trial
# SVD -> qual a dimensão da matriz (min. 62)

# %% Imports
import numpy as np
import matplotlib.pyplot as plt
import mne
from scipy.io import loadmat

# %% Loading Data

mat = loadmat('/Users/brunoandrynascimentocouto/Documents/Data/EEG/SHAM (Milano)/MR_Session_1_MotorL_TMS_EVOKED.mat')
data = np.transpose(mat['Y'], (2, 0, 1))
times = mat['times']
chlocs_file = os.path.join('/Users/chlocs_brainamp62milano.mat')


# Duas colunas:
# Labels Originais / Labels Novos
# Ler o arquivo de labels originais

# ICA: Aumentar o número de componentes

# Dicionario Milão
# AF7 AFz
# AF8 FCz
# Montagem para o standard_1020