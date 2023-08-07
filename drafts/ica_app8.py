from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QStackedWidget, QLabel, QListWidget, QListWidgetItem, QLineEdit
from PyQt5.QtCore import pyqtSlot, Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import sys
import numpy as np

class MyApp(QWidget):
    def __init__(self, data):
        super().__init__()

        self.data = data
        self.returnValue = None
        self.initUI()

    def initUI(self):
        main_layout = QHBoxLayout()

        # Left side layout
        left_layout = QVBoxLayout()

        self.good_label = QLabel('Good')
        self.good_label.setAlignment(Qt.AlignCenter)
        self.bad_label = QLabel('Bad')
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
        plot = self.create_plot()
        page_layout.addWidget(label)
        page_layout.addWidget(plot)
        page.setLayout(page_layout)
        self.stacked_widget.addWidget(page)

        # Components
        for item in self.data:
            page = QWidget()
            page_layout = QVBoxLayout()
            label = QLabel('Component ' + item)
            label.setAlignment(Qt.AlignCenter)
            plot = self.create_plot()
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
                self.stacked_widget.setCurrentIndex(index)
                self.page_number_input.setText(str(index))
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
    def create_plot(self):
        fig = Figure(figsize=(5, 4), dpi=100)
        t = np.arange(0.0, 3.0, 0.01)
        s = 1 + np.sin(2 * np.pi * t)
        fig.add_subplot(111).plot(t, s)

        canvas = FigureCanvas(fig)
        return canvas

    def closeEvent(self, event):  
        self.returnValue = "Returned Value"
        event.accept()

def ICApp8(data):
    app = QApplication(sys.argv)

    ex = MyApp(data)
    ex.show()

    try:
        app.exec_()
    except SystemExit:
        pass
    return ex.returnValue
