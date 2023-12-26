import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np


possible_status = ['AddWeight',
                   'Normal',
                   'PressureGain_constant',
                   'PropellerDamage_bad',
                   'PropellerDamage_slight']

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
        if user_input == 'x':
            break


for root, dirs, files in os.walk('Dataset'):
    # Check if we are in a directory containing CSV files
    if any(file.lower().endswith('.csv') for file in files):
        # Process each CSV file in the current directory
        for file in files:
            if file.lower().endswith('.csv'):
                file_path = os.path.join(root, file)
                label_name = root.split('/')[2]
                if label_name in possible_status:
                    label = possible_status.index(label_name)
                else:
                    label = 100
                    print('Label not found')

                AW_Data = pd.read_csv(str(file_path), header=None)
                AW_Data = np.array(AW_Data)  # Convert to numpy format
                data_visualization(AW_Data, file)