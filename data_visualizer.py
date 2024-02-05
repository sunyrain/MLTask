import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import seaborn as sns

possible_status = ['AddWeight',
                   'Normal',
                   'PressureGain_constant',
                   'PropellerDamage_bad',
                   'PropellerDamage_slight']


def intercept_datapoints(input_data):
    if input_data[:, 1].shape[0] >= 180:    # Determine if the collected data points exceed 180
        output_data = input_data[0:180, :]     # Intercept the first 180 values and discard the header
        output_data = normal(output_data)   # Standardize the signal
        return output_data
    else:
        return None


def normal(input_data: np.array):
    max_values = [1, 1200, 1200, 1200, 1200, 0.5, 700, 12, 15, 10, 200, 20, 10, 10, 10, 30, 40]
    input_data = input_data.astype(np.float32)
    for id in range(len(max_values) - 1):
        input_data[:, id] = input_data[:, id] / max_values[id]
    return input_data


def data_visualization(input_data):
    #t = input_data[:, 0]
    #data_numerical = input_data.drop(0).astype(float)
    #input_data.iloc[1:] = input_data.iloc[1:].astype(float)
    pairplot = sns.pairplot(input_data)
    fig = pairplot.fig
    fig.savefig("pairplot.png")
    # for i in range(input_data.shape[1]-1):
    #     y = input_data[:, i+1]
    #     #plt.plot(t[1:], y[1:])
    #     plt.plot(y[1:])
    #     plt.xlabel(t[0])
    #     plt.ylabel(y[0])
    #     plt.title(file_name)
    #     plt.grid()
    #     plt.show()
    #     user_input = input('Enter x to cancel:')
    #     if user_input == 'x':
    #         break


file_path = 'Dataset/train/AddWeight/AddWeight_0.csv'

data_csv = pd.read_csv(str(file_path))
data = np.array(data_csv)  # Convert to numpy format

output = intercept_datapoints(data)
if output is not None:
    output = normal(output)
    t = np.arange(1, 181)
    output[:, 0] = t
    df = pd.DataFrame(output, columns=['time', 'pwm1', 'pwm2', 'pwm3', 'pwm4',
                                       'depth', 'press', 'voltage', 'roll', 'pitch', 'yaw', 'a_x', 'a_y',
                                       'a_z', 'w_row', 'w_pitch', 'w_yaw'])
    data_visualization(df)
else:
    print("Dataset " + file_path + " < 180 datapoints --> won't be included")
