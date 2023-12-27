import numpy as np
import pandas as pd
import os

possible_status = ['AddWeight',
                   'Normal',
                   'PressureGain_constant',
                   'PropellerDamage_bad',
                   'PropellerDamage_slight']


def intercept_datapoints(input_data):
    if input_data[:, 1].shape[0] >= 180:    # Determine if the collected data points exceed 180
        output_data = input_data[0:180, 1:]     # Intercept the first 180 values and discard the header
        output_data = normal(output_data)   # Standardize the signal
        return output_data
    else:
        return None


def normal(input_data: np.array):
    max_values = [1200, 1200, 1200, 1200, 0.5, 700, 12, 15, 10, 200, 20, 10, 10, 10, 30, 40]
    input_data = input_data.astype(np.float32)
    for id in range(len(max_values)):
        input_data[:, id] = input_data[:, id] / max_values[id]
    return input_data


def label_dataset(input_data: np.array, label: int):
    label_column = np.full((input_data.shape[0], 1), label)
    new_array = np.column_stack((input_data, label_column))
    return new_array


def create_dataset(input_path: str):
    set_name = input_path.split('/')[-1]
    print('Loading ' + set_name + ' dataset')
    final_dataset = np.empty((180, 17))
    # Iterate through the directory structure
    counter = 0
    files_counter = 0
    used_files = 0
    x = 0
    unused_files = 0
    for root, dirs, files in os.walk(input_path):
        files_counter = files_counter + len(files)
        if any(file.lower().endswith('.csv') for file in files):
            for file in files:
                x = x + 1
                if file.lower().endswith('.csv'):
                    file_path = os.path.join(root, file)
                    label_name = root.split('/')[-1]
                    if label_name in possible_status:
                        label = int(possible_status.index(label_name))
                    else:
                        label = 999
                        print('Label not found')

                    data_csv = pd.read_csv(str(file_path))
                    data = np.array(data_csv)  # Convert to numpy format
                    if len(data[1]) != 17:
                        print('Corrupted input')
                    output = intercept_datapoints(data)

                    if output is not None:
                        if not np.any(output):
                            print(output)
                            print(file)
                        output = normal(output)
                        output_labelled = label_dataset(output, label)
                        final_dataset = np.vstack((final_dataset, output_labelled))
                        counter = counter + 180
                        used_files = used_files + 1
                    else:
                        print("Dataset " + file + " < 180 datapoints --> won't be included")
                        unused_files = unused_files + 1
    output_path = set_name + '.npy'
    final_dataset = final_dataset[180:, :]
    print(used_files)
    print(unused_files)
    #np.save(output_path, final_dataset)



create_dataset('Dataset/train')
create_dataset('Dataset/test')


