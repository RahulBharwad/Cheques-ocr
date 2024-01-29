import cv2
import os

class Cropper:
    def __init__(self, image_path):
        self.image_path = image_path
        self.start_point = None
        self.end_point = None

    def display_image(self):
        # Read the image
        self.image = cv2.imread(self.image_path)
        cv2.imshow("Select Region", self.image)
        cv2.waitKey(1)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.start_point is None:
                self.start_point = (x, y)
                print(f"Starting Point: {self.start_point}")
            elif self.end_point is None:
                self.end_point = (x, y)
                print(f"Ending Point: {self.end_point}")
            else:
                self.start_point = None  # Reset start_point for MICR Number
                self.end_point = None # Reset end_point for MICR Number


    def get_cropping_coordinates(self, param=None):
        # Create a window and set the callback function
        cv2.namedWindow("Select Region")
        cv2.setMouseCallback("Select Region", self.mouse_callback, param)

        while True:
            self.display_image()
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # 'Esc' key
                break

        # Destroy the OpenCV window
        cv2.destroyAllWindows()

        # Return the coordinates
        return self.start_point, self.end_point