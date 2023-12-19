import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


status = ['AddWeight', 'Normal', 'PressureGain_constant', 'PropellerDamage_bad', 'PropellerDamage_slight']


def data_visualization(input_data, file_name):
    t = input_data[:, 0]
    for i in range(input_data.shape[1]-1):
        y = input_data[:, i+1]
        #plt.plot(t[1:], y[1:])
        plt.plot(y[1:])
        plt.xlabel(t[0])
        plt.ylabel(y[0])
        plt.title(file_name)
        plt.grid()
        plt.show()
        user_input = input('Enter x to cancel:')
        #if user_input == 'x':
        #    break



def intercept_datapoints(input_data):
    if input_data[:,1].shape[0] >= 181:    # Determine if the collected data points exceed 180
        output_data = input_data[1:181,1:]     # Intercept the first 180 values and discard the header
        output_data = normal(output_data)   # Standardize the signal
        return output_data
    else:
        return None


def normal(input_data):
    max_values = [1200, 1200, 1200, 1200, 0.5, 700, 12, 15, 10, 200, 20, 10, 10, 10, 30, 40]
    input_data = input_data.astype(np.float32)
    for id in range(16):
        input_data[:, id] = input_data[:, id] / max_values[id]
    return input_data


# Iterate through the directory structure
for root, dirs, files in os.walk('Dataset'):

    # Check if we are in a directory containing CSV files
    if any(file.lower().endswith('.csv') for file in files):
        # Process each CSV file in the current directory
        for file in files:
            if file.lower().endswith('.csv'):
                file_path = os.path.join(root, file)
                if not os.path.exists(os.path.join('dataset_preprocessed', root)):
                    os.makedirs(os.path.join('dataset_preprocessed', root))

                AW_Data = pd.read_csv(str(file_path), header=None)
                AW_Data = np.array(AW_Data)  # Convert to numpy format
                data_visualization(AW_Data, file)

                output = intercept_datapoints(AW_Data)
                if output is not None:
                    output = normal(output)
                    output_path = os.path.join('dataset_preprocessed', os.path.splitext(file_path)[0] + '.npy')
                    #np.save(output_path, output)
                    #data_visualization(output)
                else:
                    print("Dataset < 180 datapoints --> won't be included")


