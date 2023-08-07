from PyQt5 import QtGui
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QStackedWidget, QLabel, QListWidget, QListWidgetItem, QLineEdit
from PyQt5.QtCore import pyqtSlot, Qt, QThread, pyqtSignal

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme('paper', 'whitegrid', 'pastel')

import sys
import numpy as np
import mne
from mne.time_frequency import psd_array_multitaper

class PlotThread(QThread):
    plot_created = pyqtSignal(Figure)

    def __init__(self, i, app, comp):
        super().__init__()
        self.i = i
        self.app = app
        self.comp = comp

    def run(self):
        component_epochs = self.app.source_data[:, self.comp, :]

        fig = Figure(figsize=(15, 15), layout='tight')
        fig.patch.set_facecolor(self.app.bg_color)
        axs = fig.subplots(3,2)

        # Evoked Signal (Original)
        self.app.evoked.plot(axes=axs[0,0], show=False)
        axs[0,0].set_title('Evoked Signal')

        # Signal with ICA Component Removed
        if self.comp not in self.app.exclude:
            self.app.ica.exclude = self.app.exclude + [self.comp]
        
        cleaned_epochs = self.app.ica.apply(self.app.epochs.copy(), exclude=self.app.ica.exclude, verbose=False)
        cleaned_evoked = cleaned_epochs.average()
        cleaned_evoked.plot(axes=axs[0,1], show=False)
        axs[0,1].set_title('Evoked Signal with Component ' + str(self.comp+1).zfill(2) + ' Removed')

        # Trials/Epochs
        im = axs[1,0].imshow(component_epochs, aspect='auto', cmap='jet')
        axs[1,0].set_title('Component Activity')
        # axs[1,0].set_xlabel('Time (s)')
        axs[1,0].set_xticks([self.app.epochs.time_as_index(0)[0]], [''])
        axs[1,0].set_ylabel('Trial')
        # fig.colorbar(im, ax=axs[1,0], label='Amplitude')

        # Average
        mean_activity = np.mean(component_epochs, axis=0)
        axs[2,0].plot(self.app.epochs.times, mean_activity)
        axs[2,0].set_xlim([self.app.epochs.times[0], self.app.epochs.times[-1]])
        axs[2,0].set_title('Average Activity')
        axs[2,0].set_xlabel('Time (s)')
        axs[2,0].set_ylabel('Amplitude')

        # PSD
        #cleaned_epochs.plot_psd(axes=axs[1,1], method='welch', verbose=False, average=True, spatial_colors=True, show=False);
        axs[1,1].psd(mean_activity, Fs=self.app.epochs.info['sfreq'])
        axs[1,1].set_xlim([0, int(self.app.epochs.info['sfreq'])/2])
        axs[1,1].set_title('Power Spectrum')

        # Topo
        self.app.ica.plot_components([self.comp], axes=axs[2,1], cmap='jet', title='', show=False)
        axs[2,1].set_title('Topography', color=self.app.text_color)

        self.plot_created.emit(fig)

class MyApp(QWidget):
    def __init__(self, ica, epochs):
        super().__init__()

        # Inputs:
        self.ica = ica.copy()
        self.epochs = epochs
        self.evoked = epochs.average()

        self.exclude = np.sort(self.ica.exclude).tolist()
        self.n_components = self.ica.n_components_
        self.ica_labels = ['ICA' + str(i).zfill(3) for i in range(self.n_components)]

        self.returnValue = None

        # Get the source signals for all components
        source_epochs = self.ica.get_sources(epochs)
        self.source_data = source_epochs.get_data()

        # Get the explained variance
        unmixing_matrix = self.ica.unmixing_matrix_
        explained_variance = np.var(unmixing_matrix, axis=1)
        explained_variance_ratio = explained_variance / np.sum(explained_variance)
        self.explained_variance_percentage = explained_variance_ratio * 100

        # Initialize the current page and canvas
        self.plots = [None] * self.n_components
        self.current_canvas = None
        self.loading_labels = []

        # Get the background color of the widget
        palette = QApplication.palette()
        bg_color = palette.color(QtGui.QPalette.Background)
        self.bg_color = bg_color.name()
        text_color = palette.color(QtGui.QPalette.WindowText)
        self.text_color = text_color.name()

        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': 'Arial',
            'font.size': 8,
            'axes.titlesize': 8,
            'axes.labelsize': 6,
            'xtick.labelsize': 5,
            'ytick.labelsize': 5,
            'text.color': self.text_color,
            'axes.labelcolor': self.text_color,
            'axes.edgecolor': self.text_color,
            'xtick.color': self.text_color,
            'ytick.color': self.text_color,
            'figure.facecolor': self.bg_color,
            'axes.facecolor': self.bg_color})

        self.initUI()

        # Make it pop up
        self.show()
        self.activateWindow()
        self.raise_()
        self.setWindowState(Qt.WindowActive)

    def initUI(self):
        main_layout = QHBoxLayout()

        # Left side layout
        left_layout = QVBoxLayout()

        self.good_label = QLabel('Use Components')
        self.good_label.setAlignment(Qt.AlignCenter)
        self.bad_label = QLabel('Remove Components')
        self.bad_label.setAlignment(Qt.AlignCenter)

        self.list1 = QListWidget()
        self.list2 = QListWidget()

        # Enable sorting
        self.list1.setSortingEnabled(True)
        self.list2.setSortingEnabled(True)

        # Connect itemDoubleClicked signal to slots
        self.list1.itemDoubleClicked.connect(self.move_item_to_list2)
        self.list2.itemDoubleClicked.connect(self.move_item_to_list1)

        self.show_button = QPushButton('Show', self)
        self.show_button.clicked.connect(self.show_item)

        self.home_button = QPushButton('Home', self)  # Home button
        self.home_button.clicked.connect(self.go_home)

        # Adding items to list
        for item in self.ica_labels:
            list_item1 = QListWidgetItem(item)
            list_item1.setTextAlignment(Qt.AlignCenter)
            self.list1.addItem(list_item1)

        button_layout = QHBoxLayout()  # Button layout for 'Home' and 'Show'
        button_layout.addWidget(self.home_button)
        button_layout.addWidget(self.show_button)

        left_layout.addWidget(self.good_label)
        left_layout.addWidget(self.list1)
        left_layout.addLayout(button_layout)
        left_layout.addWidget(self.bad_label)
        left_layout.addWidget(self.list2)

        # Create watermark
        watermark = QLabel("Couto, B.A.N. ICApp (2023).", self)
        watermark.setStyleSheet("color: rgba(200, 200, 200, 128)")
        watermark.setAlignment(Qt.AlignRight | Qt.AlignBottom)
        left_layout.addWidget(watermark)

        # Right side layout
        right_layout = QVBoxLayout()
        self.stacked_widget = QStackedWidget()

        # Create buttons
        self.button_left = QPushButton('<', self)
        self.button_right = QPushButton('>', self)

        # Create number input
        self.page_number_input = QLineEdit(self)
        self.page_number_input.setFixedWidth(50)
        self.page_number_input.setAlignment(Qt.AlignCenter)
        self.page_number_input.returnPressed.connect(self.go_to_page)

        # Connect buttons to their respective slots
        self.button_left.clicked.connect(self.go_left)
        self.button_right.clicked.connect(self.go_right)

        # Add buttons to layout
        nav_button_layout = QHBoxLayout()
        nav_button_layout.addWidget(self.button_left)
        nav_button_layout.addWidget(self.page_number_input)
        nav_button_layout.addWidget(self.button_right)
        right_layout.addLayout(nav_button_layout)

        # Create pages
        # Overview
        page = QWidget()
        page_layout = QVBoxLayout()
        label = QLabel('Overview')
        label.setAlignment(Qt.AlignCenter)
        plot = self.create_overview_plot()
        page_layout.addWidget(label)
        page_layout.addWidget(plot)
        page.setLayout(page_layout)
        self.stacked_widget.addWidget(page)

        for i, item in enumerate(self.ica_labels):
            page = QWidget()
            page_layout = QVBoxLayout()
            label = QLabel(item + f' ({self.explained_variance_percentage[i]:.2f}%)')
            label.setAlignment(Qt.AlignCenter)
            loading_label = QLabel('Loading, please wait...')
            loading_label.setAlignment(Qt.AlignCenter)
            page_layout.addWidget(label)
            page_layout.addWidget(loading_label)
            self.loading_labels.append(loading_label)

            page.setLayout(page_layout)
            self.stacked_widget.addWidget(page)


        # Add stacked widget to layout
        right_layout.addWidget(self.stacked_widget)

        # Add layouts to main layout
        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)

        # Set the layout
        self.setLayout(main_layout)

        # Move existing components to the bad list
        self.exclude_items()

    def exclude_items(self):
        # iterate over the items in the 'good' list in reverse order
        for i in range(self.list1.count()-1, -1, -1):
            item = self.list1.item(i)

            # if the item's text is in the exclude list, move it to the 'bad' list
            if int(item.text()[-3:]) in self.exclude:
                self.list1.takeItem(i)
                list_item = QListWidgetItem(item.text())
                list_item.setTextAlignment(Qt.AlignCenter)
                self.list2.addItem(list_item)

        # re-sort the lists
        self.list1.sortItems()
        self.list2.sortItems()

    def go_home(self):
        self.stacked_widget.setCurrentIndex(0)
        self.page_number_input.setText('Home')

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

    def move_item_to_list2(self):
        current_item = self.list1.currentItem()
        if current_item is not None:
            self.list1.takeItem(self.list1.row(current_item))
            self.list2.addItem(current_item)

    def move_item_to_list1(self):
        current_item = self.list2.currentItem()
        if current_item is not None:
            self.list2.takeItem(self.list2.row(current_item))
            self.list1.addItem(current_item)

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
            self.switch_to_page(page_number+1)

    def switch_to_page(self, i):
        self.stacked_widget.setCurrentIndex(i)
        self.page_number_input.setText(str(i-1))
        self.switch_page_and_plot(i)

    def switch_page_and_plot(self, i):
        if i == 0:
            self.current_canvas = None
            return

        # If the plot for this page hasn't been created yet, create it
        if self.plots[i-1] is None:
            self.plot_thread = PlotThread(i, self, i-1)
            self.plot_thread.plot_created.connect(self.on_plot_created)
            self.plot_thread.start()
        else:
            self.current_page = i
            self.current_canvas = self.plots[i-1]

    @pyqtSlot(Figure)
    def on_plot_created(self, fig):
        idx = self.stacked_widget.currentIndex()

        canvas = FigureCanvas(fig)

        # Hide loading
        loading_label = self.loading_labels[idx-1]
        loading_label.hide()

        # Adding widget
        self.plots[idx-1] = canvas
        self.stacked_widget.widget(idx).layout().addWidget(self.plots[idx-1])
        self.current_page = idx
        self.current_canvas = self.plots[idx-1]
    
    def create_overview_plot(self):
        def find_cols_and_rows(n_components):
            # Calculate the square root of the number of components to get an approximation of the grid size
            grid_size = np.sqrt(n_components)

            # Determine the number of rows and columns
            n_rows = np.floor(grid_size)
            n_cols = np.ceil(n_components / n_rows)

            # If the total slots in the grid is less than the components number, increment the rows count
            if n_rows * n_cols < n_components:
                n_rows += 1
            return int(n_rows), int(n_cols)
        
        fig = Figure(figsize=(10, 10), dpi=250)
        fig.patch.set_facecolor(self.bg_color)
        nrows, ncols = find_cols_and_rows(self.n_components)
        axs = fig.subplots(nrows, ncols)
        for i, ax in enumerate(axs.flat):
            self.ica.plot_components([i], axes=ax, cmap='jet', title='', show=False)
            ax.set_title(self.ica_labels[i], fontsize=4)

        canvas = FigureCanvas(fig)     
        return canvas

    def closeEvent(self, event):
        bad_items = []
        for index in range(self.list2.count()):
            bad_items.append(self.list2.item(index).text())
        self.ica.exclude = [int(bad_item[-3:]) for bad_item in bad_items]
        self.returnValue = self.ica
        event.accept()

qt_app = None
def ICApp(ica, epochs):
    global qt_app

    if qt_app is None:
        qt_app = QApplication(sys.argv)

    ex = MyApp(ica, epochs)
    ex.show()

    try:
        qt_app.exec_()
    except SystemExit:
        pass
    return ex.returnValue