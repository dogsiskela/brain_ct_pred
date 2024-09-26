import os
import cv2
import pydicom
import numpy as np
from pydicom.pixel_data_handlers.util import apply_voi_lut

def dcm_to_rgb_array(file_path, voi_lut = True, fix_monochrome = True):
    # Read the DICOM file
    ds = pydicom.dcmread(file_path)

    # # Get the pixel data
    # pixel_array = ds.pixel_array

    if(voi_lut):
        pixel_array = apply_voi_lut(ds.pixel_array, ds)
    else:
        pixel_array = ds.pixel_array

    if fix_monochrome and ds.PhotometricInterpretation == "MONOCHROME1":
        pixel_array = np.amax(pixel_array) - pixel_array

    # level = ds.WindowCenter
    # window = ds.WindowWidth

    # vmin = level - window/2
    # vmax = level + window/2

    # Normalize the pixel values to [0, 1]
    pixel_array -= np.min(pixel_array)  # Shift values to start from 0
    pixel_array /= np.max(pixel_array)  # Scale to [0, 1]
    pixel_array = pixel_array.astype(np.float32) # Convert to 8bit integer

    rgb_image = cv2.cvtColor(pixel_array, cv2.COLOR_BGR2RGB)

    # Resize the image to (128, 128)
    resized_image = cv2.resize(rgb_image, (128, 128))

    # Convert to a float array in the range [0, 1]
    img_array = resized_image.astype(np.float32)

    # hu_image = convert_to_hu(ds)
    
    # max_val = max(hu_image[0])
    # min_val = min(hu_image[0])

    # for i in range(1, len(hu_image)):
    #     new_max = max(hu_image[i])
    #     new_min = min(hu_image[i])
    #     if new_max>max_val:
    #         max_val = new_max
    #     if new_min<min_val:
    #         min_val = new_min

    # # print(max_val, min_val)

    # clipped_hu = np.clip(hu_image, -2048, 5036)
    # normalized_array = (clipped_hu + 2048) / 7084

    # print(max(max(normalized_array)))

    # return normalized_array, vmin, vmax
    return img_array

# def convert_to_hu(dicom_file):
#     bias = dicom_file.RescaleIntercept
#     slope = dicom_file.RescaleSlope
#     pixel_values = dicom_file.pixel_array
#     new_pixel_values = (pixel_values * slope) + bias
#     return new_pixel_values

def count_files_in_directory(directory):
    # Get a list of files in the directory
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    return len(files)
