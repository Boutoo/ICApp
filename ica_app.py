# Author: Couto, B.A.N.
# Date: August 2023
# ICA Application for MNE-Python

# %% Imports
import mne
import numpy as np
import sys

# Plots
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
sns.set_theme('paper', 'whitegrid', 'dark')

# Qt5 Imports
from PyQt5 import QtGui
from PyQt5.QtWidgets import QApplication, QWidget, QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QStackedWidget, QLabel, QListWidget, QListWidgetItem, QLineEdit, QShortcut, QFileDialog
from PyQt5.QtCore import Qt, QThread, pyqtSignal

# %% Applications Classes
qt_app = None # Global variable to store the Qt Application
class ICAWorkerThread(QThread):
    finished_signal = pyqtSignal(dict)

    def __init__(self, app):
        super().__init__()
        self.app = app

    def run(self):
        # Get Index
        index = self.app.stacked_widget.currentIndex()
        # print(f'Running thread ({index})')

        # Get the current figure and canvas
        fig = self.app.figures[index]

        if index == 0:
            self.plot_overview(fig)
        else:
            self.plot_component(fig, index-1)

        # emit the signal
        data = {
            'done': True
        }

        self.finished_signal.emit(data)

    def plot_overview(self, fig):
        if self.app.figure_is_empty[0]:
            components = self.app.ica.get_components()
            bad_channels = self.app.epochs.info['bads']
            ch_names = self.app.epochs.ch_names
            ch_names = [ch for ch in ch_names if ch not in bad_channels]
            new_info = mne.create_info(ch_names, self.app.epochs.info['sfreq'], ch_types='eeg')
            new_info.set_montage(self.app.epochs.get_montage())

            sources = self.app.source_data

            def optimal_subplot_grid(N):
                            # Find the square root of N to start approximating the grid
                            sqrt_N = np.sqrt(N)

                            # If N is a perfect square, use that as the grid dimensions
                            if sqrt_N == int(sqrt_N):
                                return int(sqrt_N), int(sqrt_N)

                            # Get the column count by rounding up the square root
                            cols = np.ceil(sqrt_N)

                            # Compute the required rows given the column count
                            rows = N // cols
                            if N % cols != 0:
                                rows += 1

                            return int(rows), int(cols)
            rows, cols = optimal_subplot_grid(self.app.n_components)
            
            if rows > 5:
                rows = 5
                cols = int(np.ceil(self.app.n_components / rows))

            gs = gridspec.GridSpec(3 * rows, cols, fig)  # 3 rows per component

            for i in range(self.app.n_components):
                row_idx = i // cols
                col_idx = i % cols
                
                # Create the topomap axis using GridSpec
                ax_topo = fig.add_subplot(gs[3*row_idx:3*row_idx+2, col_idx])
                mne.viz.plot_topomap(components[:, i],
                                     new_info,
                                     axes=ax_topo,
                                     cmap=self.app.parameters['cmap'],
                                     show=False)
                
                color = self.app.text_color
                if i in self.app.exclude:
                    color = 'red'
                ax_topo.set_title(self.app.ica_labels[i],
                             fontsize=8,
                             color=color)

                # Create the time series axis right below the topomap using GridSpec
                ax_time = fig.add_subplot(gs[3*row_idx+2, col_idx])
                
                # Plot your time series data here. For the sake of this example, I'm using random data
                component_epochs = sources[:, i, :]
                mean_activity = np.mean(component_epochs, axis=0)
                ax_time.axvline(0, color='k', linestyle='--', alpha=0.5)  # Add a vertical line at time 0
                ax_time.plot(self.app.epochs.times, mean_activity)
                t0, t1 = self.app.parameters['overview_avg_xlim'][0], self.app.parameters['overview_avg_xlim'][1]
                t0 = [self.app.epochs.times[0] if t0 is None else self.app.epochs.times[self.app.epochs.time_as_index(t0)]][0]
                t1 = [self.app.epochs.times[-1] if t1 is None else self.app.epochs.times[self.app.epochs.time_as_index(t1)]][0]
                ax_time.set_xlim([t0, t1])  # Set your x-limits
                ax_time.set_xticks([],[])
                ax_time.set_yticks([],[])
                ax_time.set_facecolor((1,1,1,self.app.parameters['bg_alpha']))  # Set the background color to white

            self.app.figure_is_empty[0] = False

        else:
            axs = fig.axes
            for i in range(self.app.n_components):
                color = self.app.text_color
                if i in self.app.ica.exclude:
                    color = 'red'
                axs[i*2].set_title(self.app.ica_labels[i],
                                   fontsize=8,
                                   color=color)
        return

    def plot_component(self, fig, comp):
        if self.app.figure_is_empty[comp+1]:
            # Clear Current Figure
            fig.clf()

            # Get data
            epochs = self.app.epochs.copy()
            ica = self.app.ica.copy()
            sources = self.app.source_data

            # Get the component epochs
            component_epochs = sources[:, comp, :]

            # Initializing Figure
            gs = fig.add_gridspec(6, 2)
            axs = [
                fig.add_subplot(gs[0:2, 0]), # Evoked
                fig.add_subplot(gs[0:2, 1]), # Evoked - Component
                fig.add_subplot(gs[2:5, 0]), # Trials
                fig.add_subplot(gs[2:4, 1]), # PSD
                fig.add_subplot(gs[5:6, 0]), # Average
                fig.add_subplot(gs[4:6, 1]), # Topo
            ]

            # Trials/Epochs
            im = axs[2].imshow(component_epochs,
                               aspect='auto',
                               cmap=self.app.parameters['cmap'])
            axs[2].set_title('Component Activity')
            axs[2].set_xticks([epochs.time_as_index(0)[0]], [''])
            axs[2].set_ylabel('Trial')

            # PSD
            f1, f2 = self.app.parameters['psd_xlim']
            if f1 is None:
                f1 = 1
            if f2 is None:
                f2 = 50
            spec, freqs = mne.time_frequency.psd_array_multitaper(component_epochs, sfreq=epochs.info['sfreq'], fmin=f1, fmax=f2, verbose=False)
            spec = 10 * np.log10(spec)
            axs[3].plot(freqs, np.mean(spec, axis=0), linewidth=1.5)
            axs[3].set_xlim([f1, f2])
            axs[3].set_title('Power Spectrum')
            axs[3].set_xlabel('Frequency (Hz)')
            axs[3].set_ylabel('Power (dB)')

            # Average
            mean_activity = np.mean(component_epochs, axis=0)
            axs[4].plot(epochs.times, mean_activity, linewidth=1.5)
            axs[4].set_xlim([epochs.times[0], epochs.times[-1]])
            axs[4].set_title('Average Activity')
            axs[4].set_xlabel('Time (s)')
            axs[4].set_ylabel('Amplitude')

            # Topo
            ica.plot_components([comp],
                                axes=axs[5],
                                cmap=self.app.parameters['cmap'],
                                title='',
                                # image_interp='nearest',
                                show=False)
            axs[5].set_title('Topography')

            # Noting that the figure is not empty anymore
            self.app.figure_is_empty[comp+1] = False
        
        # Dataset Evoked Signal (Original)-(Droped)
        if self.app.changing_component is not None or not np.any(self.app.dataset_is_updated): # Update Dataset Evoked Signal if Needed 
            self.app.clear_epochs, self.app.clear_var = self.apply_dropping()
            self.app.dataset_is_updated = [False] * (self.app.n_components)
            self.app.component_is_updated = [False] * (self.app.n_components)
            if self.app.changing_component == comp:
                self.app.component_is_updated[comp] = True
        
        if not self.app.dataset_is_updated[comp]:
            ax = fig.axes[0]
            ax.clear()
            # Display Dataset Evoked Signal
            if self.app.parameters['interactive_butterfly']:
                self.app.clear_epochs.average().plot(axes=ax, show=False)
            else:
                # ax.plot(self.app.clear_epochs.times, self.app.clear_epochs.average().data.T, linewidth=1.5)
                # ax.set_xlim([self.app.clear_epochs.times[0], self.app.clear_epochs.times[-1]])
                self.app.clear_epochs.average().plot(axes=ax, show=False, selectable=False)
            ax.set_title(f'Dataset ({self.app.clear_var:.2f}%)')

        # Signal with Current ICA Component Removed
        if not self.app.component_is_updated[comp]:
            new_epochs, new_var = self.apply_dropping(comp)
            ax = fig.axes[1]
            ax.clear()
            if self.app.parameters['interactive_butterfly']:
                new_epochs.average().plot(axes=ax, show=False)
            else:
                # ax.plot(new_epochs.times, new_epochs.average().data.T, linewidth=1.5)
                # ax.set_xlim([new_epochs.times[0], new_epochs.times[-1]])
                new_epochs.average().plot(axes=ax, show=False, selectable=False)
            ax.set_title(f'Dataset - ICA{str(comp).zfill(3)} ({new_var:.2f}%)')
        
        self.app.changing_component = None
        self.app.dataset_is_updated[comp] = True
        self.app.component_is_updated[comp] = True
        return

    def apply_dropping(self, new_exclude = None):
        # Get data
        epochs = self.app.epochs.copy()
        ica = self.app.ica.copy()

        if new_exclude is not None:
            ica.exclude += [new_exclude]

        # Calculate New Explained Variance
        ica.apply(epochs, verbose=False)
        epochs.apply_baseline(verbose=None) # ICA can introduce DC shifts
        after_removal = np.sum(epochs.get_data()**2)

        var = 100 * (after_removal/self.app.original_explained_variance)

        return epochs, var

class ICABlockingDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setModal(True)
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Processing... Please wait."))
        self.setLayout(layout)

class ICA_Application(QWidget):
    def __init__(self, ica, epochs,
                 cmap='jet',
                 apply_baseline = True,
                 psd_xlim = [None, None],
                 interactive_butterfly = True,
                 overview_avg_xlim = [None, None],
                 bg_alpha=1):
        super().__init__()
        self.setWindowTitle('ICApp')

        # Main Inputs:
        self.ica = ica.copy()
        self.epochs = epochs
        self.epochs.pick('eeg')
        if apply_baseline:
            self.epochs.apply_baseline(verbose=None)

        # Get Main Parameters:
        self.exclude = np.sort(self.ica.exclude).tolist()
        self.n_components = self.ica.n_components_
        self.ica_labels = ['Component ' + str(i).zfill(3) for i in range(1, self.n_components+1)]
        self.original_explained_variance = np.sum(epochs.get_data()**2)

        # Return value
        self.returnValue = None

        # User Parameters
        self.parameters = {
            'cmap': cmap,
            'psd_xlim': psd_xlim,
            'interactive_butterfly': interactive_butterfly,
            'bg_alpha': bg_alpha,
            'overview_avg_xlim': overview_avg_xlim,
        }

        # Get the source signals for all components
        source_epochs = self.ica.get_sources(epochs)
        self.source_data = source_epochs.get_data()

        # Plot Control
        self.dataset_is_updated = [False] * (self.n_components)
        self.component_is_updated = [False] * (self.n_components)
        self.changing_component = None

        # Setting Parameters to Plot Styles and Colors
        self.plot_style_and_colors()

        # Initializing UI
        self.initUI()
        self.figure_is_empty = [True] * (self.n_components + 1)

        # Move existing components to the bad list
        self.exclude_items()

        # Blocking Dialog
        self.dialog = ICABlockingDialog(self)

        # Worker Thread
        self.thread = ICAWorkerThread(self)
        self.thread.finished_signal.connect(self.update_plot)
        self.thread.finished_signal.connect(self.dialog.close)

        # Make UI Pop UP
        self.show()
        self.activateWindow()
        self.raise_()
        self.setWindowState(Qt.WindowActive)
        self.showMaximized()
        
        # Request Update
        self.request_update()
        
    def initUI(self):
        # Setting up layouts
        main_layout = QHBoxLayout()
        left_layout = self.setup_left_layout()
        right_layout = self.setup_right_layout()

        # Adding layouts to main layout
        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)

        # Set the layout
        self.setLayout(main_layout)

        # Set buttons starting state
        self.button_left.setEnabled(False)
        self.button_left.setText('Overview')
        self.button_right.setText('Single Component Vizualization')
        self.button_right.setToolTip('Go to single component vizualization')

    def setup_left_layout(self):
        left_layout = QVBoxLayout()

        # Labels
        self.good_label = self.create_centered_label('Kept Components')
        self.bad_label = self.create_centered_label('Removed Components')

        # List widgets
        self.list1 = self.create_list_widget(self.move_item_to_list2)
        self.list2 = self.create_list_widget(self.move_item_to_list1)
        for item in self.ica_labels:
            list_item1 = QListWidgetItem(item)
            list_item1.setTextAlignment(Qt.AlignCenter)
            self.list1.addItem(list_item1)

        # Buttons
        self.change_component_button = self.create_button('Remove/Restore [R]', self.change_item, 'R', 'Drop/Restore the selected component\n(Shortcut: R)')
        self.home_button = self.create_button('Home [H]', self.go_home, 'H', 'Go to the overview page\n(Shortcut: H)')
        self.show_button = self.create_button('Show [S]', self.show_item, 'S', 'Show the selected component\n(Shortcut: S)')
        self.save_ica_button = self.create_button('Save ICA', self.save_ica, 'Ctrl+S', 'Save the ICA object\n(Shortcut: Ctrl+S)')
        self.save_figure_button = self.create_button('Save Figure', self.save_figure, 'Ctrl+Shift+S', 'Save the current figure\n(Shortcut: Ctrl+Shift+S)')

        # Watermark
        watermark = QLabel("Couto, B.A.N. ICApp (2023).", self)
        watermark.setStyleSheet("color: rgba(200, 200, 200, 128); font-size: 8px;")
        watermark.setAlignment(Qt.AlignCenter | Qt.AlignBottom)

        # Add widgets to left layout
        widgets = [
            self.good_label,
            self.list1,
            self.bad_label,
            self.list2,
            self.change_component_button,
            self.home_button,
            self.show_button,
            self.save_ica_button,
            self.save_figure_button,
            watermark]
        
        for w in widgets:
            if isinstance(w, QWidget):
                w.setMaximumWidth(150)
                left_layout.addWidget(w)
            else:
                left_layout.addLayout(w)

        return left_layout

    def setup_right_layout(self):
        right_layout = QVBoxLayout()

        # Navigation buttons
        self.button_left = self.create_button('<', self.go_left, QtGui.QKeySequence(Qt.CTRL + Qt.Key_Left), 'Go to the previous page\n(Shortcut: Ctrl+Left)')
        self.button_right = self.create_button('>', self.go_right, QtGui.QKeySequence(Qt.CTRL + Qt.Key_Right), 'Go to the next page\n(Shortcut: Ctrl+Right)')

        # Number input
        self.page_number_input = QLineEdit(self)
        self.page_number_input.setFixedWidth(50)
        self.page_number_input.setAlignment(Qt.AlignCenter)
        self.page_number_input.returnPressed.connect(self.go_to_page)

        # Add navigation widgets to layout
        nav_button_layout = QHBoxLayout()
        for widget in [self.button_left, self.page_number_input, self.button_right]:
            nav_button_layout.addWidget(widget)
        right_layout.addLayout(nav_button_layout)

        # Pages
        self.setup_pages()

        # Add stacked widget to layout
        right_layout.addWidget(self.stacked_widget)

        return right_layout

    def setup_pages(self):
        self.stacked_widget = QStackedWidget()
        self.labels = []
        self.figures = []
        self.canvases = []

        # Overview page
        page, label, fig, canvas = self.create_page('Overview')
        self.labels.append(label)
        self.figures.append(fig)
        self.canvases.append(canvas)
        self.stacked_widget.addWidget(page)

        # Component pages
        for item in self.ica_labels:
            page, label, fig, canvas = self.create_page(item)
            self.labels.append(label)
            self.figures.append(fig)
            self.canvases.append(canvas)
            self.stacked_widget.addWidget(page)

    def create_page(self, title):
        page = QWidget()
        page_layout = QVBoxLayout()
        label = self.create_centered_label(title)
        fig = Figure(figsize=(10, 10), layout='constrained')
        canvas = FigureCanvas(fig)
        page_layout.addWidget(label)
        page_layout.addWidget(canvas)
        page.setLayout(page_layout)
        return page, label, fig, canvas

    def create_button(self, text, slot, shortcut=None, tooltip=None):
        btn = QPushButton(text, self)
        btn.clicked.connect(slot)
        if shortcut is not None:
            QShortcut(QtGui.QKeySequence(shortcut), self).activated.connect(slot)
        if tooltip is not None:
            btn.setToolTip(tooltip)
        return btn

    def create_centered_label(self, text):
        label = QLabel(text)
        label.setAlignment(Qt.AlignCenter)
        return label

    def create_list_widget(self, double_click_slot):
        list_widget = QListWidget()
        list_widget.setSortingEnabled(True)
        list_widget.itemDoubleClicked.connect(double_click_slot)
        return list_widget

    def go_home(self):
        self.page_number_input.setText('Home')
        self.switch_to_page(0)

    def show_item(self):
        current_widget = QApplication.focusWidget()
        current_item = None

        if current_widget in [self.list1, self.list2] and current_widget.selectedItems():
            current_item = current_widget.selectedItems()[0]

        if current_item is not None:
            try:
                index = self.ica_labels.index(current_item.text())
                self.switch_to_page(index + 1)
            except ValueError:
                pass

    def change_item(self):
        current_widget = QApplication.focusWidget()
        current_item = None

        if current_widget in [self.list1, self.list2] and current_widget.selectedItems():
            current_item = current_widget.selectedItems()[0]

        if current_item is not None:
            if current_widget == self.list1:
                self.move_item_to_list2()
            else:
                self.move_item_to_list1()

    def move_item_to_list2(self):
        current_item = self.list1.currentItem()
        if current_item is not None:
            self.list1.takeItem(self.list1.row(current_item))
            self.list2.addItem(current_item)
            self.ica.exclude = self.get_bads()
            self.changing_component = int(current_item.text()[-3:])-1
            self.request_update()

    def move_item_to_list1(self):
        current_item = self.list2.currentItem()
        if current_item is not None:
            self.list2.takeItem(self.list2.row(current_item))
            self.list1.addItem(current_item)
            self.ica.exclude = self.get_bads()
            self.changing_component = int(current_item.text()[-3:])-1
            self.request_update()

    def go_left(self):
        current_index = self.stacked_widget.currentIndex()
        if current_index > 0:
            self.switch_to_page(current_index - 1)

    def go_right(self):
        current_index = self.stacked_widget.currentIndex()
        if current_index < self.stacked_widget.count() - 1:
            self.switch_to_page(current_index + 1)

    def go_to_page(self):
        page_number = self.page_number_input.text()
        if not page_number.isdigit():
            return
        
        page_number = int(page_number)

        if 0 <= page_number < self.n_components:
            self.switch_to_page(page_number)

    def switch_to_page(self, i):
        self.stacked_widget.setCurrentIndex(i)
        self.page_number_input.setText(str(i))
        self.current_page = i
        if i == 0:
            self.button_left.setEnabled(False)
            self.button_left.setText('Overview')
            self.button_right.setText('Single Component Vizualization')
            self.button_right.setToolTip('Go to single component vizualization')
        elif i==1:
            self.button_left.setEnabled(True)
            self.button_left.setText('Overview')
            self.button_left.setToolTip('Go to the overview component')
            self.button_right.setEnabled(True)
            self.button_right.setText('Next >')
            self.button_right.setToolTip('Go to the next component')
        elif i==self.n_components:
            self.button_left.setEnabled(True)
            self.button_left.setText('< Previous')
            self.button_left.setToolTip('Go to the previous component')
            self.button_right.setEnabled(False)
            self.button_right.setText('Last Component')
        else:
            self.button_left.setEnabled(True)
            self.button_left.setText('< Previous')
            self.button_left.setToolTip('Go to the previous component')
            self.button_right.setEnabled(True)
            self.button_right.setText('Next >')
            self.button_right.setToolTip('Go to the next component')
        self.request_update()

    def request_update(self):
        # Get Index
        # index = self.stacked_widget.currentIndex()
        # print(f'Requesting update ({index})')

        self.thread.start()
        self.dialog.exec_()

    def update_plot(self):
        # Get Index
        index = self.stacked_widget.currentIndex()
        # print(f'Updating plot ({index})')
        canvas = self.canvases[index]
        canvas.draw()

    def get_bads(self):
        bad_items = []
        for index in range(self.list2.count()):
            bad_items.append(self.list2.item(index).text())
        bads = [int(bad_item[-3:])-1 for bad_item in bad_items]
        return bads

    def exclude_items(self):
        # iterate over the items in the 'good' list in reverse order
        for i in range(self.list1.count()-1, -1, -1):
            item = self.list1.item(i)

            # if the item's text is in the exclude list, move it to the 'bad' list
            if int(item.text()[-3:])-1 in self.exclude:
                self.list1.takeItem(i)
                list_item = QListWidgetItem(item.text())
                list_item.setTextAlignment(Qt.AlignCenter)
                self.list2.addItem(list_item)

        # re-sort the lists
        self.list1.sortItems()
        self.list2.sortItems()

    def plot_style_and_colors(self):
        palette = QApplication.palette()
        bg_color = palette.color(QtGui.QPalette.Background)
        self.bg_color = bg_color.name()
        text_color = palette.color(QtGui.QPalette.WindowText)
        self.text_color = text_color.name()

        # Plot Visual Parameters
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': 'Arial',
            'font.size': 8,
            'axes.titlesize': 10,
            'axes.labelsize': 8,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'text.color': self.text_color,
            'axes.labelcolor': self.text_color,
            # 'axes.edgecolor': (self.text_color),
            # 'xtick.color': self.text_color,
            # 'ytick.color': self.text_color,
            'figure.facecolor': self.bg_color,
            'axes.facecolor': (1,1,1,self.parameters['bg_alpha'])})

    def save_figure(self):
        # Get Index
        index = self.stacked_widget.currentIndex()
        # print(f'Saving figure ({index})')

        # Get the current figure and canvas
        fig = self.figures[index]

        # Save the figure
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getSaveFileName(self, "Save Figure", "", "PNG Files (*.png);;JPEG Files (*.jpg);; SVG Files (*.svg);;All Files (*)", options=options)
        if fileName:
            fig.savefig(fileName)

    def save_ica(self):
        # Save the figure
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getSaveFileName(self, "Save ICA", "", "ICA Files (*-ica.fif);;;;All Files (*)", options=options)
        if fileName:
            if not fileName.endswith('-ica.fif'):
                fileName += '-ica.fif'
            self.ica.save(fileName, overwrite=True)

    def closeEvent(self, event):
        plt.close('all')
        self.ica.exclude = self.get_bads()
        self.returnValue = self.ica
        event.accept()

# %% Application Calls:
def ICApp(ica, epochs,
          cmap='turbo',
          apply_baseline = True,
          psd_xlim = [None, None],
          interactive_butterfly = False,
          overview_avg_xlim = [None, None]):
    global qt_app

    if qt_app is None:
        qt_app = QApplication(sys.argv)

    print('Running ICApp...')
    ex = ICA_Application(ica, epochs,
                         cmap=cmap,
                         apply_baseline=apply_baseline,
                         psd_xlim=psd_xlim,
                         interactive_butterfly=interactive_butterfly,
                         overview_avg_xlim=overview_avg_xlim)
    ex.show()

    try:
        qt_app.exec_()
    except SystemExit:
        pass
    return ex.returnValue
