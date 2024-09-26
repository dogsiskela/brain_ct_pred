from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from data_processing import dcm_to_rgb_array, count_files_in_directory


# pd_new = pd.read_pickle('./patient_data.pkl')


def get_brain_array_from_patient_data():
    brain_imgs = []
    files_count = count_files_in_directory('brain_ct/data/patient_1/')

    for i in range(0, files_count):
        array = dcm_to_rgb_array('brain_ct/data/patient_1/slice_'+str(i)+'.dcm')
        brain_imgs.append(array)

    image = 401
    plt.imshow(brain_imgs[image], cmap='gray', alpha=1)
    plt.show()
    return np.array([np.array(el).astype(np.float32) for el in brain_imgs]).astype(np.float32)

print(get_brain_array_from_patient_data()[100])
