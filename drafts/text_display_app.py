from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel
import sys

def text_app(values):
    app = QApplication(sys.argv)

    window = QWidget()
    layout = QVBoxLayout()

    for value in values:
        layout.addWidget(QLabel(str(value)))

    window.setLayout(layout)
    window.show()
    
    try:
        sys.exit(app.exec_())
    except SystemExit:
        pass