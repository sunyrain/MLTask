import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import OneHotEncoder


possible_status = ['AddWeight',
                   'Normal',
                   'PressureGain_constant',
                   'PropellerDamage_bad',
                   'PropellerDamage_slight']


def intercept_datapoints(input_data: np.array) -> np.array:
    if input_data[:, 1].shape[0] >= 180:    # Determine if the collected data points exceed 180
        output_data = input_data[0:180, :]     # Intercept the first 180 values
        return output_data
    else:
        return None


def normal(input_data: np.array) -> np.array:
    max_values = [1200, 1200, 1200, 1200, 0.5, 700, 12, 15, 10, 200, 20, 10, 10, 10, 30, 40]
    input_data = input_data.astype(np.float32)
    for id in range(len(max_values)):
        input_data[:, id] = input_data[:, id] / max_values[id]
    return input_data


def create_dataset(input_path: str):
    set_name = input_path.split('/')[-1]
    print('Loading ' + set_name + ' dataset')

    files_counter = 0
    unused_files = 0
    preprocessed_data_list = []
    label_list = []
    # Iterate through the directory structure
    for root, dirs, files in os.walk(input_path):
        files_counter = files_counter + len(files)
        if any(file.lower().endswith('.csv') for file in files):
            for file in files:
                if file.lower().endswith('.csv'):
                    file_path = os.path.join(root, file)
                    label_name = root.split('/')[-1]
                    if label_name in possible_status:
                        label = int(possible_status.index(label_name))
                    else:
                        label = 999
                        print('Label not found')
                    data_csv = pd.read_csv(str(file_path))
                    data = np.array(data_csv)
                    if len(data[1]) != 17:
                        print('Corrupted input')
                    output = intercept_datapoints(data)
                    if output is not None:
                        output = normal(output)
                        preprocessed_data_list.append(output.flatten())
                        label_list.append(label)
                    else:
                        print("Dataset " + file + " < 180 datapoints --> won't be included")
                        unused_files = unused_files + 1

    preprocessed_data_array = np.array(preprocessed_data_list)
    label_array = np.array(label_list).reshape(-1, 1).flatten()
    #label_array = OneHotEncoder(sparse_output=False).fit_transform(label_array)
    preprocessed_data_output_path = 'data/X_' + set_name + '.npy'
    label_array_path = 'data/y_' + set_name + '.npy'

    print('Found ' + str(files_counter) + ' files,' + ' unused files: ' + str(unused_files))

    np.save(preprocessed_data_output_path, preprocessed_data_array)
    np.save(label_array_path, label_array)


create_dataset('Dataset/train')
create_dataset('Dataset/test')


