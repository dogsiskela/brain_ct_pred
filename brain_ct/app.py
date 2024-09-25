
# import matplotlib
# matplotlib.use("TkAgg")

import numpy as np
from src.models.models import predict
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QApplication, QComboBox, QGridLayout, QListWidget, QPushButton, QLabel, QWidget, QSlider
from PyQt5.QtGui import QIcon
import matplotlib.pyplot as plt
import pandas as pd
import sys
from src.data.data_test import get_brain_array_from_patient_data

OPEN_PARAGRAPH = "<p>"
CLOSE_PARAGRAPH = "</p>"
OPEN_HEADER = "<h4>"
CLOSE_HEADER = "</h4>"


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None):
        fig, self.ax = plt.subplots(figsize=(5, 4), dpi=100)
        super().__init__(fig)

    def plot_multiple(self, images, alphas):
        # Clear previous plots
        self.ax.clear()

        # Plot each image with corresponding alpha, overlapping
        for image, alpha in zip(images, alphas):
            self.ax.imshow(image, cmap='Reds', alpha=alpha)

        self.ax.axis('off')  # Hide axes
        self.draw()  # Render the figure


class exoplanetFilter(QWidget):
    def __init__(self, parent=None):
        super(exoplanetFilter, self).__init__(parent)

        self.patient_id = 49
        self.predictions = []

        # new test
        self.new_data = get_brain_array_from_patient_data()

        pd_new = pd.read_pickle('./patient_data.pkl')
        pd_new = pd_new[pd_new['PatientNumber'] == int(self.patient_id)]

        self.patient_data = pd_new

        layout = QGridLayout()
        self.index = 0

        self.labMission = QLabel('<h4>Image:</h4>')

        self.brain_image_label = QLabel('<h4>Brain image:</h4>')
        self.mask_original_label = QLabel('<h4>Mask original:</h4>')
        self.mask_predicted_label = QLabel('<h4>Mask predicted:</h4>')

        # TCE Type container
        self.labStarType = QLabel(
            OPEN_PARAGRAPH + 'Patients:' + CLOSE_PARAGRAPH)
        self.patient_label = QLabel(
            OPEN_PARAGRAPH + 'Patient Number:' + CLOSE_PARAGRAPH)
        self.patient_number_box = QComboBox()
        self.patient_number_box.addItems([str(i) for i in range(49, 130)])
        self.patient_number_box.currentIndexChanged.connect(
            self.changePatientNumber)

        # Mission container
        self.missionCb = QComboBox()
        self.missionCb.addItems([str(i) for i in range(0, 100)])
        self.missionCb.currentIndexChanged.connect(self.tempFunc)
        self.listWidget = QListWidget()
        self.listWidget.itemClicked.connect(self.tempFunc)

        self.listWidget.addItems(['test'])

        # Buttons
        self.create_prediction_button = QPushButton("Create prediction")
        self.create_prediction_button.clicked.connect(self.buttonClick)

        #
        self.condition_label = QLabel(OPEN_HEADER + "Condition:" +
                                      CLOSE_HEADER + OPEN_PARAGRAPH + str(self.patient_data.iloc[self.index]['Condition on file']) + CLOSE_PARAGRAPH)

        # S tar metadata
        self.patient_number_label = QLabel(
            OPEN_HEADER + "Patient Number:" + CLOSE_HEADER + OPEN_PARAGRAPH + str(self.patient_data.iloc[self.index]['PatientNumber']) + CLOSE_PARAGRAPH)
        self.has_hemorrhage_label = QLabel(
            OPEN_HEADER + "Has Hemmorrhage:" + CLOSE_HEADER + OPEN_PARAGRAPH + str(self.patient_data.iloc[self.index]['Has_Hemorrhage']) + CLOSE_PARAGRAPH)
        self.has_fracture_label = QLabel(
            OPEN_HEADER + "Has fracture:" + CLOSE_HEADER + OPEN_PARAGRAPH + str(self.patient_data.iloc[self.index]['Fracture_Yes_No']) + CLOSE_PARAGRAPH)

        # Create canvas for plot previews
        self.canvas = MplCanvas(self)
        self.canvas_mask = MplCanvas(self)
        self.canvas_predictions = MplCanvas(self)

        # Create a slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(int(len(self.patient_data))-1)
        self.slider.setValue(0)  # Set default value
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval(10)

        # Slider label
        self.label = QLabel("Value: 0")

        # On slider value changed
        self.slider.valueChanged.connect(self.update_label)

        # Build widget layout
        layout.addWidget(self.slider, 0, 0)
        layout.addWidget(self.label, 0, 1)
        layout.addWidget(self.patient_label, 0, 2)
        layout.addWidget(self.patient_number_box, 0, 3)
        layout.addWidget(self.brain_image_label, 1, 0, 1, 1)
        layout.addWidget(self.mask_original_label, 1, 1, 1, 2)
        layout.addWidget(self.mask_predicted_label, 1, 3, 1, 5)
        layout.addWidget(self.canvas, 2, 0, 1, 1)
        layout.addWidget(self.canvas_mask, 2, 1, 1, 2)
        layout.addWidget(self.canvas_predictions, 2, 3, 1, 5)
        layout.addWidget(self.patient_number_label, 3, 1)
        layout.addWidget(self.condition_label, 3, 2)
        layout.addWidget(self.has_hemorrhage_label, 3, 3)
        layout.addWidget(self.has_fracture_label, 3, 4)
        layout.addWidget(self.create_prediction_button, 4, 0)
        prediction = self.patient_data.iloc[self.index]['ImagePathBrain']
        prediction2 = self.patient_data.iloc[self.index]['ImagePathBrainMask']

        self.canvas.plot_multiple([prediction], [1])
        self.canvas_mask.plot_multiple([prediction, prediction2], [1, 0.7])

        if len(self.predictions) > 0:
            prediction3 = self.predictions[self.index]
            self.canvas_predictions.plot_multiple(
                [prediction, prediction3], [1, 0.7])
        else:
            self.canvas_predictions.plot_multiple([prediction], [1])

        # Set the layout
        self.setLayout(layout)
        self.setWindowTitle("Brain CT scans")

    def tempFunc(self):
        pass

    def buttonClick(self):
        predictions = predict(
            np.array([np.array(el).astype(np.float32) for el in self.patient_data['ImagePathBrain']]).astype(np.float32))
        self.predictions = predictions
        prediction3 = predictions[self.index]
        self.canvas_predictions.plot_multiple(
            [self.patient_data.iloc[self.index]['ImagePathBrain'], prediction3], [1, 0.7])

    def changePatientNumber(self):
        curr_val = self.patient_number_box.currentText()
        if (curr_val == ''):
            return
        pd_new = pd.read_pickle('./patient_data.pkl')
        pd_new = pd_new[pd_new['PatientNumber'] == int(curr_val)]
        self.slider.setValue(0)
        self.slider.setMaximum(int(len(pd_new))-1)
        self.patient_data = pd_new
        self.predictions = []
        self.update_image(0)

    def update_label(self):
        # Update label with the current value of the slider
        curr_val = self.slider.value()
        self.label.setText(f"Value: {curr_val}")

        self.update_image(curr_val)

    def update_image(self, curr_val):
        self.index = int(curr_val)
        prediction = self.patient_data.iloc[self.index]['ImagePathBrain']
        prediction2 = self.patient_data.iloc[self.index]['ImagePathBrainMask']
        self.canvas.plot_multiple([prediction], [1])
        self.canvas_mask.plot_multiple([prediction, prediction2], [1, 0.7])
        if len(self.predictions) > 0:
            prediction3 = self.predictions[self.index]
            self.canvas_predictions.plot_multiple(
                [prediction, prediction3], [1, 0.7])
        else:
            self.canvas_predictions.plot_multiple([prediction], [1])

        self.condition_label.setText(OPEN_HEADER + "Condition:" +
                                     CLOSE_HEADER + OPEN_PARAGRAPH + str(self.patient_data.iloc[self.index]['Condition on file']) + CLOSE_PARAGRAPH)
        self.patient_number_label.setText(
            OPEN_HEADER + "Patient Number:" + CLOSE_HEADER + OPEN_PARAGRAPH + str(self.patient_data.iloc[self.index]['PatientNumber']) + CLOSE_PARAGRAPH)
        self.has_hemorrhage_label.setText(
            OPEN_HEADER + "Has Hemmorrhage:" + CLOSE_HEADER + OPEN_PARAGRAPH + str(self.patient_data.iloc[self.index]['Has_Hemorrhage']) + CLOSE_PARAGRAPH)
        self.has_fracture_label.setText(
            OPEN_HEADER + "Has fracture:" + CLOSE_HEADER + OPEN_PARAGRAPH + str(self.patient_data.iloc[self.index]['Fracture_Yes_No']) + CLOSE_PARAGRAPH)


def start_app():
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon('./brain.png'))
    ex = exoplanetFilter()
    ex.show()
    sys.exit(app.exec_())


start_app()
