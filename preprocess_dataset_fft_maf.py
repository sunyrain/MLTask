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


def FFTvibData(vib_value):
    fft_freq = np.fft.fftfreq(len(vib_value))
    fft_result = np.fft.fft(vib_value)
    fft_result_shifted = np.fft.fftshift(fft_result)
    fft_freq_shifted = np.fft.fftshift(fft_freq)

    #vib_fft = [np.fft.fft(vib_value[i, :]) for i in range(vib_value.shape[0])]
    vib_fft_abs = np.abs(fft_result_shifted)
    vib_fft_ang = np.angle(fft_result_shifted)

    return vib_fft_abs, vib_fft_ang, fft_freq_shifted


def timeWindows(vib_fft, length=10):

    vib_fft_TW = []
    for i in range(vib_fft.shape[1]):
        temp = vib_fft[:length, i].tolist()
        for j in range(vib_fft.shape[0] - length):
            a = np.mean(vib_fft[j:(j + length), i])
            temp.append(a)
        vib_fft_TW.append(temp)

    vib_fft_TW = np.array(vib_fft_TW).T

    return vib_fft_TW


def ifftPredict(vib_fft_pre, select_fre, NUM=500):
    vib_fft_pre = abs(vib_fft_pre) * np.exp(1j * np.random.uniform(0, 2 * np.pi, (vib_fft_pre.shape)))
    vib_fft = np.zeros([vib_fft_pre.shape[0], NUM], dtype=complex)
    vib_fft[:, select_fre] = vib_fft_pre
    select_fre = list(NUM - np.array(select_fre))
    select_fre.remove(NUM)
    vib_fft_pre = vib_fft_pre.real - 1j * vib_fft_pre.imag
    vib_fft[:, select_fre] = vib_fft_pre[:, 1:]
    vib_pre = np.array([np.fft.ifft(vib_fft[i, :]).real for i in range(vib_fft_pre.shape[0])]).reshape(-1)

    return vib_pre


def ifftOrigin(vib_fft_abs, vib_fft_ang):
    vib_fft = abs(vib_fft_abs) * np.exp(1j * vib_fft_ang)
    vib_origin = np.array([np.fft.ifft(vib_fft[i, :]).real for i in range(vib_fft.shape[0])]).reshape(-1)

    return (vib_origin)


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

                        select_fre = range(0, 200)
                        vib_abs_1, vib_ang_1, freq = FFTvibData(output)
                        #vib_1_ary = vib_abs_1[:, select_fre]
                        vib_1_ary = timeWindows(vib_abs_1, length=10)

                        preprocessed_data_list.append(vib_1_ary.flatten())
                        label_list.append(label)
                    else:
                        print("Dataset " + file + " < 180 datapoints --> won't be included")
                        unused_files = unused_files + 1

    preprocessed_data_array = np.array(preprocessed_data_list)
    label_array = np.array(label_list).reshape(-1, 1).flatten()
    #label_array = OneHotEncoder(sparse_output=False).fit_transform(label_array)
    preprocessed_data_output_path = 'data/X_' + set_name + '_fft.npy'
    label_array_path = 'data/y_' + set_name + '_fft.npy'

    print('Found ' + str(files_counter) + ' files,' + ' unused files: ' + str(unused_files))

    np.save(preprocessed_data_output_path, preprocessed_data_array)
    np.save(label_array_path, label_array)


create_dataset('Dataset/train')
create_dataset('Dataset/test')


