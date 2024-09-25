import os
import cv2
import pydicom
import numpy as np

def dcm_to_rgb_array(file_path):
    # Read the DICOM file
    ds = pydicom.dcmread(file_path)

    # Get the pixel data
    pixel_array = ds.pixel_array.astype(np.float32)

    # Normalize the pixel values to [0, 1]
    pixel_array -= np.min(pixel_array)  # Shift values to start from 0
    pixel_array /= np.max(pixel_array)  # Scale to [0, 1]

    rgb_image = cv2.cvtColor(pixel_array, cv2.COLOR_BGR2RGB)

    # Resize the image to (128, 128)
    resized_image = cv2.resize(rgb_image, (128, 128))

    # Convert to a float array in the range [0, 1]
    # img_array = resized_image.astype(np.float32)

    return img_array

def count_files_in_directory(directory):
    # Get a list of files in the directory
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    return len(files)
