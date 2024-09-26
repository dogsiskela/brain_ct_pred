import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from src.data.data_processing import dcm_to_rgb_array, count_files_in_directory


# pd_new = pd.read_pickle('./patient_data.pkl')


def get_brain_array_from_patient_data(loc='brain_ct/data/Xyqfg/XKG9Y'):
    brain_imgs = []
    files_count = count_files_in_directory(loc)

    for i in range(0, files_count):
        array = dcm_to_rgb_array(loc+'/slice_'+str(i)+'.dcm')
        brain_imgs.append(array)

    return np.array([np.array(el).astype(np.float32) for el in brain_imgs]).astype(np.float32)


def get_folders_with_files():
    folders_with_files = []

    for root, dirs, files in os.walk('brain_ct/data'):
        if files:
            folders_with_files.append(root)

    return folders_with_files
