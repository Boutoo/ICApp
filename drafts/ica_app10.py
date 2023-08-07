from PyQt5 import QtGui
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QStackedWidget, QLabel, QListWidget, QListWidgetItem, QLineEdit
from PyQt5.QtCore import pyqtSlot, Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme('paper', 'whitegrid', 'pastel')

import sys
import numpy as np
import mne
from mne.time_frequency import psd_array_multitaper


class MyApp(QWidget):
    def __init__(self, ica, epochs):
        super().__init__()

        # Inputs:
        self.ica = ica.copy()
        self.epochs = epochs
        self.evoked = epochs.average()

        self.exclude = []
        self.data = [str(i+1).zfill(2) for i in range(ica.n_components_)]
        self.returnValue = None

        # Get the source signals for all components
        source_epochs = ica.get_sources(epochs)
        self.source_data = source_epochs.get_data()

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

        # Adding items to list for demonstration, you can add your own items
        for item in self.data:
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
        watermark = QLabel("Couto, B.A.N. (2023). ICApp.", self)
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

        # Components
        for i, item in enumerate(self.data):
            page = QWidget()
            page_layout = QVBoxLayout()
            label = QLabel('ICA ' + item)
            label.setAlignment(Qt.AlignCenter)
            plot = self.create_ica_plot(i)
            page_layout.addWidget(label)
            page_layout.addWidget(plot)
            page.setLayout(page_layout)
            self.stacked_widget.addWidget(page)

        # Add stacked widget to layout
        right_layout.addWidget(self.stacked_widget)

        # Add layouts to main layout
        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)

        # Set the layout
        self.setLayout(main_layout)

    def go_home(self):
        self.stacked_widget.setCurrentIndex(0)
        self.page_number_input.setText('0')

    def show_item(self):
        current_widget = QApplication.focusWidget()
        current_item = None

        if current_widget in [self.list1, self.list2] and current_widget.selectedItems():
            current_item = current_widget.selectedItems()[0]

        if current_item is not None:
            try:
                index = self.data.index(current_item.text())
                self.stacked_widget.setCurrentIndex(index+1)
                self.page_number_input.setText(str(index+1))
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
            self.stacked_widget.setCurrentIndex(current_index - 1)
            self.page_number_input.setText(str(current_index - 1))

    def go_right(self):
        current_index = self.stacked_widget.currentIndex()
        if current_index < self.stacked_widget.count() - 1:
            self.stacked_widget.setCurrentIndex(current_index + 1)
            self.page_number_input.setText(str(current_index + 1))

    def go_to_page(self):
        page_number = int(self.page_number_input.text())
        if 0 <= page_number < self.stacked_widget.count():
            self.stacked_widget.setCurrentIndex(page_number)

    # PLOT HERE
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
        nrows, ncols = find_cols_and_rows(len(self.data))
        axs = fig.subplots(nrows, ncols)
        for i, ax in enumerate(axs.flat):
            self.ica.plot_components([i], axes=ax, cmap='jet', title='', show=False)
            ax.set_title('ICA ' + str(i+1).zfill(2), fontsize=4)

        canvas = FigureCanvas(fig)     
        return canvas

    def create_ica_plot(self, comp):
        component_epochs = self.source_data[:, comp, :]

        fig = Figure(figsize=(15, 15), layout='tight')
        fig.patch.set_facecolor(self.bg_color)
        axs = fig.subplots(3,2)

        # Evoked Signal (Original)
        self.evoked.plot(axes=axs[0,0], show=False)
        axs[0,0].set_title('Evoked Signal')

        # Signal with ICA Component Removed
        if comp not in self.exclude:
            self.ica.exclude = self.exclude + [comp]
        
        cleaned_epochs = self.ica.apply(self.epochs.copy(), exclude=self.ica.exclude, verbose=False)
        cleaned_evoked = cleaned_epochs.average()
        cleaned_evoked.plot(axes=axs[0,1], show=False)
        axs[0,1].set_title('Evoked Signal with Component ' + str(comp+1).zfill(2) + ' Removed')

        # Trials/Epochs
        im = axs[1,0].imshow(component_epochs, aspect='auto', cmap='jet')
        axs[1,0].set_title('Component Activity')
        #axs[1,0].set_xlabel('Time (s)')
        axs[1,0].set_xticks([self.epochs.time_as_index(0)[0]], [''])
        axs[1,0].set_ylabel('Trial')
        # fig.colorbar(im, ax=axs[1,0], label='Amplitude')

        # Average
        mean_activity = np.mean(component_epochs, axis=0)
        axs[2,0].plot(self.epochs.times, mean_activity)
        axs[2,0].set_xlim([self.epochs.times[0], self.epochs.times[-1]])
        axs[2,0].set_title('Average Activity')
        axs[2,0].set_xlabel('Time (s)')
        axs[2,0].set_ylabel('Amplitude')

        # PSD
        #cleaned_epochs.plot_psd(axes=axs[1,1], method='welch', verbose=False, average=True, spatial_colors=True, show=False);
        axs[1,1].psd(mean_activity, Fs=self.epochs.info['sfreq'])
        axs[1,1].set_xlim([0, int(self.epochs.info['sfreq'])/2])
        axs[1,1].set_title('Power Spectrum')

        # Topo
        self.ica.plot_components([comp], axes=axs[2,1], cmap='jet', title='', show=False)
        axs[2,1].set_title('Topography', color=self.text_color)

        canvas = FigureCanvas(fig)
        return canvas

    def closeEvent(self, event):
        bad_items = []
        for index in range(self.list2.count()):
            bad_items.append(self.list2.item(index).text())
        self.ica.exclude = [int(bad_item) for bad_item in bad_items]
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