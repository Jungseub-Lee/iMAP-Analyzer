# main.py
import sys
from PyQt5.QtWidgets import QApplication
from src.gui import ImageProcessingApp

def main():
    """
    Entry point for iMAP Analyzer.
    Initializes the Qt Application and launches the Main Window.
    """
    app = QApplication(sys.argv)
    
    # Create and show the GUI window
    window = ImageProcessingApp()
    window.show()
    
    print("iMAP Analyzer started successfully.")
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()