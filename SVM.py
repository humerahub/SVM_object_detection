from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QPushButton,
    QLabel,
    QFileDialog,
    QScrollArea,
)
from PySide6.QtGui import QPixmap, QImage, QPainter, QPen
from PySide6.QtCore import Qt
import cv2
import numpy as np
import joblib
from skimage.feature import hog


class ImageClassifierApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.model = joblib.load("image_classifier_model.pkl")

    def initUI(self):
        layout = QVBoxLayout()

        self.upload_button = QPushButton("Upload Image")
        self.upload_button.clicked.connect(self.upload_image)
        layout.addWidget(self.upload_button)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.image_label = QLabel("No Image")
        self.scroll_area.setWidget(self.image_label)
        layout.addWidget(self.scroll_area)

        self.result_label = QLabel("Result: None")
        layout.addWidget(self.result_label)

        self.setLayout(layout)
        self.setWindowTitle("Image Classifier")
        self.show()

    def upload_image(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open Image File", "", "Image Files (*.png *.jpg *.bmp)"
        )
        if file_name:
            image = cv2.imread(file_name)
            resized_image = cv2.resize(image, (256, 256))
            features, veg_bbox, water_bbox, has_vegetation, has_water = (
                extract_hog_features(resized_image)
            )
            features = np.array(features).reshape(
                1, -1
            )  # Convert to NumPy array and reshape
            prediction = self.model.predict(features)

            result_text = "None"
            if has_vegetation and has_water:
                result_text = "Both Vegetation and Water Body"
            elif has_vegetation:
                result_text = "Vegetation"
            elif has_water:
                result_text = "Water Body"

            self.result_label.setText(f"Result: {result_text}")
            qt_image = QImage(file_name)
            pixmap = QPixmap.fromImage(qt_image)
            painter = QPainter(pixmap)

            # Red bounding box for vegetation at the top left
            pen = QPen(Qt.red)
            pen.setWidth(5)
            painter.setPen(pen)

            if veg_bbox:
                x, y, w, h = veg_bbox
                top_left_x = 10
                top_left_y = 10
                painter.drawRect(top_left_x, top_left_y, w, h)
                painter.drawText(top_left_x, top_left_y - 10, "Vegetation")

            # Green bounding box for water body at the center
            if water_bbox:
                pen.setColor(Qt.green)
                painter.setPen(pen)

                img_height, img_width = pixmap.height(), pixmap.width()
                box_size = 150  # Increased box size
                center_x, center_y = img_width // 2, img_height // 2
                new_x = max(0, center_x - box_size // 2)
                new_y = max(0, center_y - box_size // 2)
                new_w = min(box_size, img_width - new_x)
                new_h = min(box_size, img_height - new_y)
                painter.drawRect(new_x, new_y, new_w, new_h)
                painter.drawText(new_x, new_y - 10, "Water Body")

            painter.end()
            self.image_label.setPixmap(pixmap)


def extract_hog_features(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features, hog_image = hog(
        gray_image, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=True
    )

    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([140, 255, 255])
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])

    mask_water = cv2.inRange(image, lower_blue, upper_blue)
    mask_green = cv2.inRange(image, lower_green, upper_green)

    water_bbox = get_combined_bounding_box(mask_water)
    veg_bbox = get_combined_bounding_box(mask_green)

    has_vegetation = veg_bbox is not None
    has_water = water_bbox is not None

    return features, veg_bbox, water_bbox, has_vegetation, has_water


def get_combined_bounding_box(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    x_min, y_min = float("inf"), float("inf")
    x_max, y_max = float("-inf"), float("-inf")
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + w)
        y_max = max(y_max, y + h)
    return (x_min, y_min, x_max - x_min, y_max - y_min)


if __name__ == "__main__":
    app = QApplication([])
    ex = ImageClassifierApp()
    app.exec()
