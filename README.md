# ICApp

## Overview

ICApp is a Python-based application that offers a graphical user interface to view and analyze MNE Tools ICA components. It categorizes the components into 'Good' and 'Bad' lists, allows for dynamic interaction through arrow navigation and direct page jumping. The application also supports visualization of useful plots corresponding to each component. Lazy plotting is used since displaying some of the plots is computationally expensive.

## Features

- **Dynamic Navigation**: Navigate through pages using left and right arrows, or directly jump to a specific page by entering the page number or selecting the desired component and pressing 'Show'.
- **Interactive Lists**: Manage 'Good' and 'Bad' lists, allowing users to double-click to move items between lists and sort them. Highlighted items can be selected to navigate to the corresponding page.
- **Visualizations**: Generate Matplotlib plots corresponding to the elements of the input list, with support for subplots and on-demand plotting.

## Installation

### Prerequisites

Make sure you have Python (version x.x or newer) and the following libraries installed:

- mne
- PyQt5 (usually shipped with mne)
- Matplotlib (usually shipped with mne)

## Usage

To open the application window and view the values passed, you can call the app function with the required parameters:

```python
from ica_app import ICApp
epochs = mne.Epochs(...) # MNE's Epochs object
ica = mne.preprocessing.ICA(...) # MNE's ICA object

new_ica = ICApp(ica, epochs)
```

## Contributing

If you would like to contribute, please fork the repository and use a feature branch. Pull requests and issues descriptions are warmly welcome.

## License

MIT License

## Acknowledgments

This work was made in collaboration with the Neuroengineering Laboratory from the Institute of Science and Technology of the Federal University of São Paulo.
