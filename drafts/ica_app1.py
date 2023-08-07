from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QStackedWidget, QLabel
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import sys
import numpy as np

class MyApp(QWidget):
    def __init__(self, data):
        super().__init__()
        
        self.data = data
        self.return_value = None
        self.initUI()
        
    def initUI(self):
        self.layout = QVBoxLayout()
        self.stacked_widget = QStackedWidget()

        # Create buttons
        self.button_left = QPushButton('<', self)
        self.button_right = QPushButton('>', self)
        self.button_home = QPushButton('Home', self)

        # Connect buttons to their respective slots
        self.button_left.clicked.connect(self.go_left)
        self.button_right.clicked.connect(self.go_right)
        self.button_home.clicked.connect(self.go_home)

        # Add buttons to layout
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.button_left)
        button_layout.addWidget(self.button_home)
        button_layout.addWidget(self.button_right)
        self.layout.addLayout(button_layout)

        # Create pages
        for item in self.data:
            page = QWidget()
            page_layout = QVBoxLayout()
            label = QLabel(item)
            plot = self.create_plot()
            page_layout.addWidget(label)
            page_layout.addWidget(plot)
            page.setLayout(page_layout)
            self.stacked_widget.addWidget(page)

        # Add stacked widget to layout
        self.layout.addWidget(self.stacked_widget)
        
        # Set the layout
        self.setLayout(self.layout)

    def go_left(self):
        current_index = self.stacked_widget.currentIndex()
        if current_index > 0:
            self.stacked_widget.setCurrentIndex(current_index - 1)

    def go_right(self):
        current_index = self.stacked_widget.currentIndex()
        if current_index < self.stacked_widget.count() - 1:
            self.stacked_widget.setCurrentIndex(current_index + 1)

    def go_home(self):
        self.stacked_widget.setCurrentIndex(0)

    def create_plot(self):
        fig = Figure(figsize=(5, 4), dpi=100)
        t = np.arange(0.0, 10.0, 0.01)
        s = np.sin(np.random.rand()* 2 * np.pi * t)
        fig.add_subplot(111).plot(t, s)

        canvas = FigureCanvas(fig)
        return canvas
    
    def closeEvent(self, event):  # Add this method
        self.returnValue = "Returned Value"
        event.accept()

def ICApp(data):
    app = QApplication(sys.argv)

    ex = MyApp(data)
    ex.show()

    try:
        app.exec_()
    except SystemExit:
        pass
    return ex.returnValue  # Return the returnValue attribute
