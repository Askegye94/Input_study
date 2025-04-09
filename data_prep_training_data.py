# -*- coding: utf-8 -*-

## PRE STEP STUFF ##    
windowLength = 200 # do not change this.
############################
## Some required settings ##
############################
inputColumns = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
latentFeatures = 3 # 2 #  3 /     --> Sina is currently writing the paper visuals for 2 latentfeatures and validation also for 4 latentfeatures. (2 and 4?)
trainModel =  False #True #False

frequency = 50
wanted_seconds = 4
original_frequency = 150

storingWeights = False #True   # true / false
############################
##### end of settings! #####  20 * 10 seconds = input data 200  --> more gait cycles included. 
############################  50 * 4 seconds = 200 samples. 


if frequency == 20:
    timeLength = 10
    refactorValue = 5 # taking each 5th value for resampling
else:
    timeLength = 4
    refactorValue = 2 

import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import scipy.io as spio
import pandas as pd
import seaborn as sns
from scipy.interpolate import interp1d
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from scipy.stats import mode
from sklearn.preprocessing import RobustScaler


plt.close('all')

# Define the base path
base_path = r'C:\Users\gib445\OneDrive - Vrije Universiteit Amsterdam\PHD Aske\Papers\OMC_IMU_VIDEO'
pathToDataRelativeAngles = r'C:\Users\gib445\OneDrive - Vrije Universiteit Amsterdam\PHD Aske\Papers\Longitudinal study of Stroke\VAE_stroke_longitudinal\proximalangles\proximalangles\Stroke'
pathToDataEvents = r'C:\Users\gib445\OneDrive - Vrije Universiteit Amsterdam\PHD Aske\Papers\Longitudinal study of Stroke\VAE_stroke_longitudinal\Events\Events\Stroke'
pathToCleanData = os.path.join(base_path, '3dPreparedData')
pathToAnomalies = os.path.join(base_path, 'Outliers')

############### STEP 1 ###############################
filename = []
data = []
group = []

#### THE TIMESERIES #####
if 'pathToDataRelativeAngles' in locals():
    for file1 in os.listdir(pathToDataRelativeAngles):
        full_path = os.path.join(pathToDataRelativeAngles, file1)
        data.append(spio.loadmat(full_path))
        filename.append(file1)
        # Group based on numeric part in filename
        group.append(('prox_relativeangles'.join(filter(str.isdigit, file1))))

group = np.array(group)

# Extract data from the loaded .mat files
for numtrials in range(len(filename)):
    data[numtrials] = data[numtrials]['prox_relativeangles']

    # Scale the data independently for each file and column
    scaler = RobustScaler()
    
    # Apply scaling on each trial's data (assuming data[numtrials] is a 2D array where rows are samples and columns are features)
    data[numtrials] = scaler.fit_transform(data[numtrials])

    
columns_names = ['LKneeAngles_X', 'LKneeAngles_Y', 'LKneeAngles_Z',
                 'LHipAngles_X', 'LHipAngles_Y', 'LHipAngles_Z',
                 'LAnkleAngles_X', 'LAnkleAngles_Y', 'LAnkleAngles_Z',
                 'RKneeAngles_X', 'RKneeAngles_Y', 'RKneeAngles_Z',
                 'RHipAngles_X', 'RHipAngles_Y', 'RHipAngles_Z',
                 'RAnkleAngles_X', 'RAnkleAngles_Y', 'RAnkleAngles_Z']


#%%

# Define the refactor value based on the single file type's frequency
refactor_value = original_frequency / frequency  # Adjust based on your file's original frequency

# Initialize a list to store downsampled data
downsampled_data = []

# Iterate through data for downsampling
for array in data:
    num_samples = array.shape[0]
    num_columns = array.shape[1]
    new_length = int(num_samples / refactor_value)  # Calculate the new number of samples for this array

    if refactor_value != 1.0:
        new_array = np.zeros((new_length, num_columns))

        # Perform column-wise interpolation
        for col in range(num_columns):
            x_old = np.linspace(0, num_samples - 1, num_samples)
            x_new = np.linspace(0, num_samples - 1, new_length)
            f = interp1d(x_old, array[:, col], kind='linear')
            new_array[:, col] = f(x_new)

        downsampled_data.append(new_array)
    else:
        downsampled_data.append(array)  # Append unchanged data if no downsampling is needed

#%%
###### THE GAIT EVENTS ############

groupE = []
dataE = []
filenameE = []

if 'pathToDataEvents' in locals():
    for file1E in os.listdir(pathToDataEvents):
        full_pathE = os.path.join(pathToDataEvents, file1E)
        dataE.append(spio.loadmat(full_pathE))
        filenameE.append(file1E)
        # Extract numeric grouping based on the filename
        groupE.append(int(''.join(filter(str.isdigit, file1E))))

groupE = np.array(groupE)

# Extract the 'Events' data from the loaded .mat files
for numtrials in range(len(filenameE)):
    dataE[numtrials] = dataE[numtrials]['Events']  # Adjust the key if needed

############### STEP 2 ###################################
#%%

# Iterate through dataE
for indx in range(len(dataE)):
    # Use a single original frequency
    original_frequency = original_frequency  # Adjust this to the correct frequency of your data

    # Perform rounding based on the adjusted frequency ratio
    dataE[indx] = np.round(dataE[indx] * (frequency / original_frequency), 0)

#%%

# # overlapping windows
# TrialE = []
# count = 0
# trackGroup = []
# trackGroup1 = []
# dataAugmented = np.zeros((0, len(inputColumns)))  # Initialize as empty with correct number of columns
# gait_cycles = 1

# for indx in range(len(downsampled_data)):
#     if len(downsampled_data[indx]) > windowLength:
#         for indx2 in range(len(dataE[indx]) - gait_cycles):  # Loop through each index except the last one
#             current_index = int(dataE[indx][indx2][0])
#             next_index = int(dataE[indx][indx2 + gait_cycles][0])
#             tempArray = downsampled_data[indx][current_index:next_index]
#             if len(tempArray) > 0:  # Ensure there's data between the indices
#                 # Interpolate tempArray to have windowLength rows
#                 num_columns = tempArray.shape[1]
#                 new_tempArray = np.zeros((windowLength, num_columns))
                
#                 for col in range(num_columns):
#                     x_old = np.linspace(0, len(tempArray) - 1, len(tempArray))
#                     x_new = np.linspace(0, len(tempArray) - 1, windowLength)
#                     f = interp1d(x_old, tempArray[:, col], kind='linear', fill_value="extrapolate")
#                     new_tempArray[:, col] = f(x_new)
                
#                 count += 1
#                 dataAugmented = np.vstack((dataAugmented, new_tempArray[:, inputColumns]))
#                 trackGroup.append(indx)
#                 trackGroup1.append('S' + str(group[indx]))  # Updated to a single consistent prefix
#                 TrialE.append(int(''.join(filter(str.isdigit, filename[indx][-5:]))))  # Extract trial number


TrialE = []
count = 0
trackGroup = []
trackGroup1 = []
dataAugmented = np.zeros((0, len(inputColumns)))  # Initialize as empty with correct number of columns

num_frames = int(frequency * wanted_seconds)  # Calculate frames per window

for indx in range(len(downsampled_data)):
    if len(downsampled_data[indx]) > num_frames:  # Only process data with enough length
        for indx2 in range(len(dataE[indx])):  # Loop through gait events
            current_index = int(dataE[indx][indx2][0])
            next_index = current_index + num_frames  # Define the window size based on frequency * wanted_seconds

            # Ensure the window doesn't exceed the available data
            if next_index <= len(downsampled_data[indx]):
                tempArray = downsampled_data[indx][current_index:next_index]

                # Interpolate tempArray to have `windowLength` rows
                num_columns = tempArray.shape[1]
                new_tempArray = np.zeros((windowLength, num_columns))
                
                for col in range(num_columns):
                    x_old = np.linspace(0, len(tempArray) - 1, len(tempArray))
                    x_new = np.linspace(0, len(tempArray) - 1, windowLength)
                    f = interp1d(x_old, tempArray[:, col], kind='linear', fill_value="extrapolate")
                    new_tempArray[:, col] = f(x_new)
                
                # Add to dataAugmented and tracking variables
                count += 1
                dataAugmented = np.vstack((dataAugmented, new_tempArray[:, inputColumns]))
                trackGroup.append(indx)
                trackGroup1.append('S' + str(group[indx]))  # Keep consistent prefix for subject
                TrialE.append(int(''.join(filter(str.isdigit, filename[indx][-5:]))))  # Extract trial number


#%%
######## STEP 3 SPLIT THE DATASET ##################################

y = []  # Initialize y as an empty list

# Process trackGroup to extract subject ID and type of walk
for indx in range(len(trackGroup)):
    # Extract the filename for the current trackGroup index
    current_file = filename[trackGroup[indx]]
    
    # Split the filename to extract subject ID and walk type
    parts = current_file.split('_')  # Split by underscore
    subject_id = parts[0]  # The first part contains the subject ID, e.g., 'S19'
    walk_type = parts[1].replace('.mat', '')  # The second part contains the walk type, e.g., 'walk55'
    
    # Append the subject ID and walk type as a list to y
    y.append([subject_id, walk_type])

#%%
#Anomaly detection
df = pd.DataFrame(dataAugmented)
window_size = 200

anomaly_inputs_x = [0,3,6,9,12,15]
anomaly_inputs_y = [1,4,7,10,13,16]
anomaly_inputs_z = [2,5,8,11,14,17]
# all_inputs = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]

# Fit the isolation forest model
model = IsolationForest(
    n_estimators=70,
    max_samples=0.8,
    contamination=0.25,
    max_features=1,
    bootstrap=False,
    n_jobs=-1,
    random_state=42
)
model.fit(df[anomaly_inputs_x])

df['anomaly_scores'] = model.decision_function(df[anomaly_inputs_x])
df['anomaly'] = model.predict(df[anomaly_inputs_x])

num_columns = dataAugmented.shape[1]
dataAugmented = dataAugmented.reshape(len(y),window_size,num_columns)
anomalies = np.array(df)[:,num_columns:]
anomalies_augmented = anomalies.reshape(len(y),window_size,2)
data_mean = np.mean(dataAugmented, axis=1)

#%%
#plotting and calculating anomalies

def calculatemean(data, data_mean):
    # Split the array into two separate arrays
    first_column = data[:,:, 0]
    second_column = data[:,:, 1]
    # Compute the mean for the first column
    mean_first_column = np.mean(first_column, axis=1)
    # Compute the mode for the second column
    mode_second_column = mode(second_column, axis=1).mode
    # mode_second_column = np.squeeze(mode_second_column, axis=1)
    # Create a third column based on the condition
    third_column = np.where(mean_first_column >= 0, 1, -1)
    # Calculate the percentage of -1 in the third column
    percentage_minus_one = np.count_nonzero(third_column == -1) / len(third_column) * 100
    print("Percentage of -1 in the third column: {:.2f}%".format(percentage_minus_one))
    # Combine the results
    anomalies_mean = np.dstack([mean_first_column, mode_second_column, third_column])
    anomalies_mean = np.squeeze(anomalies_mean, axis=0)
    print(anomalies_mean.shape)
    data_mean_plot = np.concatenate((data_mean, anomalies_mean), axis=1)
    df = pd.DataFrame(data_mean_plot)
    df.rename(columns={18: 'anomaly_scores', 19: 'anomaly_mode', 20: 'anomaly_mean'}, inplace=True)
    return df

df = calculatemean(anomalies_augmented, data_mean)

anomalies_list = np.where(df['anomaly_mean'] == -1)[0].tolist()

# outlier_plot(df, 'Isolation Forest')
palette = ['#ff7f0e','#1f77b4']
sns.pairplot(df, vars=anomaly_inputs_x, hue='anomaly_mean', palette=palette)


#%%

def splitdata(data, outliers):
    # Convert anomalies_list to a set for faster lookup
    anomalies_set = set(outliers)

    # Create masks for clean and outlier data
    mask_outliers = np.array([i in anomalies_set for i in range(data.shape[0])])
    mask_clean = ~mask_outliers

    # Split the data into clean and outlier subsets
    data_outliers = data[mask_outliers]
    data_clean = data[mask_clean]
    
    # Print the dimensions of the subsets
    print("Dimensions of data_outliers:", data_outliers.shape)
    print("Dimensions of data_clean:", data_clean.shape)
    
    mean_clean = np.mean(data_clean, axis=0)
    std_clean = np.std(data_clean, axis=0)
    mean_outliers = np.mean(data_outliers, axis=0)
    std_outliers = np.std(data_outliers, axis=0)
    return [mean_clean, std_clean, mean_outliers, std_outliers]

plot_data = splitdata(dataAugmented, anomalies_list)
#%%
# Plotting
fig, axs = plt.subplots(3, 6, figsize=(25, 15))

# Flatten the axes array for easy iteration
axs = axs.flatten()

# Plot the data
for i in range(num_columns):
    # Mean and std for clean data
    mean_clean = plot_data[0][:, i]
    std_clean = plot_data[1][:, i]
    
    # Mean and std for outlier data
    mean_outliers = plot_data[2][:, i]
    std_outliers = plot_data[3][:, i]
    
    # Plot clean data mean with std shaded area
    axs[i].plot(mean_clean, label='Clean Mean', color='blue')
    axs[i].fill_between(range(len(mean_clean)), mean_clean - std_clean, mean_clean + std_clean, color='blue', alpha=0.3)

    # Plot outlier data mean with std shaded area
    axs[i].plot(mean_outliers, label='Outlier Mean', color='red')
    axs[i].fill_between(range(len(mean_outliers)), mean_outliers - std_outliers, mean_outliers + std_outliers, color='red', alpha=0.3)

    axs[i].set_title(f'Column {i+1}')
    axs[i].legend()

# Hide any unused subplots
for i in range(num_columns, len(axs)):
    fig.delaxes(axs[i])

plt.tight_layout()
plt.show()


#%%

externSubjects = ['', '', '', '']  # Example: ['S1', 'S38', 'S10', 'S20']
indexenexternSubjects = []
indexeninternSubjects = []
train_data = []
test_data = []
extern_data = []

## External validation handling ###
for indx in range(0, len(trackGroup1)):
    subject_id = y[indx][0]  # Get the subject ID (e.g., 'S19') from y
    if subject_id not in externSubjects:
        indexeninternSubjects.append(indx)
    else:
        indexenexternSubjects.append(indx)

# Prepare external and internal data
extern_data = dataAugmented[indexenexternSubjects, :, :]
other_data = dataAugmented[indexeninternSubjects, :, :]
y_adapted = [y[i] for i in indexeninternSubjects]  # Adapt y to match internal subjects

# Create group splits based on subjects and save walk types
subject = 0
groupsplit = [subject]
walk_names = [y[indexeninternSubjects[0]][1]]  # Start with the first walk name

for indx in range(1, len(indexeninternSubjects)):
    current_subject = y[indexeninternSubjects[indx]][0]  # Subject ID of the current index
    previous_subject = y[indexeninternSubjects[indx - 1]][0]  # Subject ID of the previous index
    
    if current_subject == previous_subject:
        groupsplit.append(subject)
    else:
        subject += 1
        groupsplit.append(subject)
        walk_names.append(y[indexeninternSubjects[indx]][1])  # Add the walk name for the new subject

groupsplit = np.array(groupsplit)


#%%
### saving other_data, y_adapted and groupsplit. so next the manuscrit can start over here!

np.save(os.path.join(pathToCleanData, f"stored_3D_other_data_stroke_latentfeatures1_{latentFeatures}_frequency_{frequency}"), other_data)
np.save(os.path.join(pathToCleanData, f"stored_y_3D_adapted_stroke_latentfeatures1_{latentFeatures}_frequency_{frequency}"), y_adapted)
np.save(os.path.join(pathToCleanData, f"stored_3D_groupsplit_stroke_latentfeatures1_{latentFeatures}_frequency_{frequency}"), groupsplit)

filename = 'exclude_files_stroke.txt'
file_path = os.path.join(pathToAnomalies, filename)

# Open the file in write mode and write the list to it
with open(file_path, 'w') as file:
    for item in anomalies_list:
        file.write(str(item) + '\n')