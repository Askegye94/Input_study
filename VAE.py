# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 17:33:49 2025

@author: gib445
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 00:50:48 2024

@author: gib445
"""

for name in dir():
    if not name.startswith('_'):
        del globals()[name]
        
#%%

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import numpy as np
import os
import random
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from tensorflow.keras import layers
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib import gridspec
import math
from scipy import stats
from scipy.stats import pearsonr
import statsmodels.api as sm
from statsmodels.formula.api import mixedlm
from scipy.stats import ttest_rel
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from scipy.stats import ttest_rel

# Configuration
seed_value = 1
data_path = r'C:\Users\gib445\OneDrive - Vrije Universiteit Amsterdam\PHD Aske\Papers\OMC_IMU_VIDEO\3dPreparedData\\'
IMU_OMC_data_path = r'C:\Users\gib445\OneDrive - Vrije Universiteit Amsterdam\PHD Aske\Papers\OMC_IMU_VIDEO\3dPreparedData'
model_save_path = r'C:\Users\gib445\OneDrive - Vrije Universiteit Amsterdam\PHD Aske\Papers\OMC_IMU_VIDEO\\'
window_length = 200
batch_size = 64
latent_features = 3  # Latent feature size
learning_rate = 3e-4  # Lower learning rate to improve stability
epochs = 10 # Number of training epochs

input1 = [20, 50]  # filters, kernel size
input2 = [10, 40]  # filters, kernel size
input3 = [2, 30]   # filters, kernel size
input4 = [80]      # units in Dense layer

columns_names = ['LKneeAngles_X', 'LKneeAngles_Y', 'LKneeAngles_Z',
                 'LHipAngles_X', 'LHipAngles_Y', 'LHipAngles_Z',
                 'LAnkleAngles_X', 'LAnkleAngles_Y', 'LAnkleAngles_Z',
                 'RKneeAngles_X', 'RKneeAngles_Y', 'RKneeAngles_Z',
                 'RHipAngles_X', 'RHipAngles_Y', 'RHipAngles_Z',
                 'RAnkleAngles_X', 'RAnkleAngles_Y', 'RAnkleAngles_Z']

# New Configuration Flags
use_saved_weights = True  # Set this to True to load saved weights, False for random initialization

# Set random seed for reproducibility
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

# Load Data
def load_data():
    data_files = {
        'groups_healthy': 'stored_3D_groupsplit_healthy_latentfeatures1_3_frequency_50.npy',
        'groups_stroke': 'stored_3D_groupsplit_stroke_latentfeatures1_3_frequency_50.npy',
        'data_healthy': 'stored_3D_other_data_healthy_latentfeatures1_3_frequency_50.npy',
        'data_stroke': 'stored_3D_other_data_stroke_latentfeatures1_3_frequency_50.npy',
        'y_healthy': 'stored_y_3D_adapted_healthy_latentfeatures1_3_frequency_50.npy',
        'y_stroke': 'stored_y_3D_adapted_stroke_latentfeatures1_3_frequency_50.npy',    
        'groups_IMU_OMC': 'stored_3D_groupsplit_OMC_latentfeatures1_3_frequency_50.npy',
        'data_IMU': 'stored_3D_other_data_IMU_latentfeatures1_3_frequency_50.npy',
        'data_OMC': 'stored_3D_other_data_OMC_latentfeatures1_3_frequency_50.npy',
        'ids_IMU_OMC': 'stored_y_3D_adapted_OMC_latentfeatures1_3_frequency_50.npy',
        'ids_IMU_only': 'stored_y_3D_adapted_IMU_latentfeatures1_3_frequency_50.npy'
    }
    groups_healthy = np.load(os.path.join(data_path, data_files['groups_healthy']))
    groups_stroke = np.load(os.path.join(data_path, data_files['groups_stroke']))
    data_healthy = np.load(os.path.join(data_path, data_files['data_healthy']))
    data_stroke = np.load(os.path.join(data_path, data_files['data_stroke']))
    y_healthy = np.load(os.path.join(data_path, data_files['y_healthy']))
    y_stroke = np.load(os.path.join(data_path, data_files['y_stroke']))
    groups_IMU_OMC = np.load(os.path.join(IMU_OMC_data_path, data_files['groups_IMU_OMC']))
    data_IMU = np.load(os.path.join(IMU_OMC_data_path, data_files['data_IMU']))
    data_OMC = np.load(os.path.join(IMU_OMC_data_path, data_files['data_OMC']))
    ids_IMU_OMC = np.load(os.path.join(IMU_OMC_data_path, data_files['ids_IMU_OMC']))
    ids_IMU_only = np.load(os.path.join(IMU_OMC_data_path, data_files['ids_IMU_only']))
    
    exclude_files_healthy = np.loadtxt(r'C:\Users\gib445\OneDrive - Vrije Universiteit Amsterdam\PHD Aske\Papers\OMC_IMU_VIDEO\Outliers\exclude_files_healthy.txt')
    exclude_files_stroke = np.loadtxt(r'C:\Users\gib445\OneDrive - Vrije Universiteit Amsterdam\PHD Aske\Papers\OMC_IMU_VIDEO\Outliers\exclude_files_stroke.txt')
    outliers_IMU_OMC = np.loadtxt(r'C:\Users\gib445\OneDrive - Vrije Universiteit Amsterdam\PHD Aske\Papers\OMC_IMU_VIDEO\Outliers\outliers_IMU_OMC.txt')
    exclude_files_healthy = np.int_(exclude_files_healthy)
    exclude_files_stroke = np.int_(exclude_files_stroke)
    exclude_outliers_IMU_OMC = np.int_(outliers_IMU_OMC)
    
    groups_healthy = np.delete(groups_healthy, exclude_files_healthy, axis=0)
    groups_stroke = np.delete(groups_stroke, exclude_files_stroke, axis=0)
    data_healthy = np.delete(data_healthy, exclude_files_healthy, axis=0)
    data_stroke = np.delete(data_stroke, exclude_files_stroke, axis=0)
    y_healthy = np.delete(y_healthy, exclude_files_healthy, axis=0)
    y_stroke = np.delete(y_stroke, exclude_files_stroke, axis=0)
    groups_IMU_OMC = np.delete(groups_IMU_OMC, exclude_outliers_IMU_OMC, axis=0)
    data_IMU = np.delete(data_IMU, exclude_outliers_IMU_OMC, axis=0)
    data_OMC = np.delete(data_OMC, exclude_outliers_IMU_OMC, axis=0)
    ids_IMU_OMC = np.delete(ids_IMU_OMC, exclude_outliers_IMU_OMC, axis=0)
    ids_IMU_only = np.delete(ids_IMU_only, exclude_outliers_IMU_OMC, axis=0)

    # Cast data to float32 to ensure consistency
    groups_healthy = groups_healthy.astype(np.float32)
    groups_stroke = groups_stroke.astype(np.float32) 
    data_healthy = data_healthy.astype(np.float32)
    data_stroke = data_stroke.astype(np.float32) 
    groups_IMU_OMC = groups_IMU_OMC.astype(np.float32)
    data_IMU = data_IMU.astype(np.float32)
    data_OMC = data_OMC.astype(np.float32)

    return groups_healthy, groups_stroke, data_healthy, data_stroke, data_IMU, data_OMC, groups_IMU_OMC, ids_IMU_OMC, ids_IMU_only, y_healthy, y_stroke

# Split Data into Training and Test Sets
def split_data(other_data, y_adapted, groupsplit, test_size=0.3):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed_value)
    train_idx, test_idx = next(gss.split(other_data, y_adapted, groups=groupsplit))
    
    X_train, X_test = other_data[train_idx], other_data[test_idx]
    y_train, y_test = y_adapted[train_idx], y_adapted[test_idx]
    groups_train, groups_test = groupsplit[train_idx], groupsplit[test_idx]
    
    return X_train, X_test, y_train, y_test, groups_train, groups_test

# Main script to load and split data
groups_healthy, groups_stroke, data_healthy, data_stroke, data_IMU, data_OMC, groups_IMU_OMC, ids_IMU_OMC, ids_IMU_only, y_healthy, y_stroke = load_data()

# Concatenate the data
stroke_healthy_data = np.concatenate((data_healthy, data_stroke), axis=0)
stroke_healthy_labels = np.concatenate((y_healthy, y_stroke), axis=0)
stroke_healthy_groups = np.concatenate((groups_healthy, groups_stroke), axis=0)

test_data = np.concatenate((data_IMU, data_OMC), axis=0)
test_labels = np.concatenate((ids_IMU_OMC, ids_IMU_OMC), axis=0)

X_train, X_test, y_train, y_test, groups_train, groups_test  = split_data(stroke_healthy_data, stroke_healthy_labels, stroke_healthy_groups)

# # Select only specific joint angle columns
# selected_indices = [0, 3, 4, 5, 6, 7, 9, 12, 13, 14, 15, 16]

# # Apply filtering to all datasets
# X_train = X_train[:, :, selected_indices]
# X_test = X_test[:, :, selected_indices]
# test_data = test_data[:, :, selected_indices]
# data_IMU = data_IMU[:, :, selected_indices]
# data_OMC = data_OMC[:, :, selected_indices]

# Print the shapes of the datasets
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
print("test_data shape:", test_data.shape)
print("test labels shape:", test_labels.shape)

# List of columns that should NOT be flipped
flipped_axis_training_test = [0, 1, 4, 7, 9, 11, 14, 17]
flipped_axis_IMU_OMC = [1, 5, 10, 14]

number_of_columns = X_train.shape[-1]

# Loop through all columns and flip only those that are in the excluded list for training/test data
for i in range(number_of_columns):
    if i in flipped_axis_training_test:
        X_train[:, :, i] *= -1
        X_test[:, :, i] *= -1
        
# Loop through all columns and flip only those that are in the excluded list for IMU/OMC data
for i in range(number_of_columns):
    if i in flipped_axis_IMU_OMC:
        data_IMU[:, :, i] *= -1
        data_OMC[:, :, i] *= -1                

# Define the RMSE and NRMSE functions
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def nrmse(predictions, targets):
    return np.sqrt((((predictions - targets) / targets) ** 2).mean())

# Define the sampling function for latent features
def sample_latent_features(distribution):
    distribution_mean, distribution_variance = distribution
    batch_size = tf.shape(distribution_variance)[0]
    random_sample = tf.keras.backend.random_normal(shape=(batch_size, tf.shape(distribution_variance)[1]))
    return distribution_mean + tf.exp(0.5 * distribution_variance) * random_sample

# Custom layer to calculate KL divergence
class KLDivergenceLayer(layers.Layer):
    def call(self, inputs):
        distribution_mean, distribution_variance = inputs
        kl_loss = 1 + distribution_variance - tf.square(distribution_mean) - tf.exp(distribution_variance)
        kl_loss = tf.reduce_mean(kl_loss) * -0.5
        return kl_loss

from keras.saving import register_keras_serializable

@register_keras_serializable()
class CVAE(tf.keras.Model):
    def __init__(self, latent_features):
        super(CVAE, self).__init__()
        self.latent_features = latent_features

        # Encoder
        self.encoder_input = tf.keras.layers.Input(shape=(window_length, number_of_columns))
        encoder = tf.keras.layers.Conv1D(filters=input1[0], kernel_size=input1[1], activation='relu')(self.encoder_input)
        encoder = tf.keras.layers.Conv1D(filters=input2[0], kernel_size=input2[1], activation='relu')(encoder)
        encoder = tf.keras.layers.Conv1D(filters=input3[0], kernel_size=input3[1], activation='relu')(encoder)
        
        encoder = tf.keras.layers.Flatten()(encoder)
        encoder = tf.keras.layers.Dense(input4[0])(encoder)

        self.mean = tf.keras.layers.Dense(latent_features, name='mean')(encoder)
        self.log_variance = tf.keras.layers.Dense(latent_features, name='log_variance')(encoder)
        self.latent_encoding = tf.keras.layers.Lambda(sample_latent_features)([self.mean, self.log_variance])
        self.encoder_model = tf.keras.Model(self.encoder_input, [self.mean, self.log_variance, self.latent_encoding])

        # Decoder
        self.decoder_input = tf.keras.layers.Input(shape=(latent_features,))
        decoder = tf.keras.layers.Dense(input4[0])(self.decoder_input)
        decoder = tf.keras.layers.Reshape((1, input4[0]))(decoder)
        decoder = tf.keras.layers.Conv1DTranspose(filters=input3[0], kernel_size=input3[1], activation='relu')(decoder)
        decoder = tf.keras.layers.Conv1DTranspose(filters=input2[0], kernel_size=input2[1], activation='relu')(decoder)
        decoder = tf.keras.layers.Conv1DTranspose(filters=input1[0], kernel_size=input1[1], activation='relu')(decoder)
        decoder_output = tf.keras.layers.Conv1DTranspose(filters=number_of_columns, kernel_size=83)(decoder)
        decoder_output = tf.keras.layers.LeakyReLU(negative_slope=0.1)(decoder_output)
        self.decoder_model = tf.keras.Model(self.decoder_input, decoder_output)

        # KL Divergence layer
        self.kl_loss_layer = KLDivergenceLayer()
        
    def encode(self, x):
        mean, logvar, _ = self.encoder_model(x)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=tf.shape(mean))
        return eps * tf.exp(logvar * 0.5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder_model(z)
        if apply_sigmoid:
            return tf.sigmoid(logits)
        return logits

# Define the loss function components
def get_loss(kl_loss_layer):
    def get_reconstruction_loss(y_true, y_pred):
        reconstruction_loss = tf.keras.losses.mse(y_true, y_pred)
        reconstruction_loss_batch = tf.reduce_mean(reconstruction_loss)
        return reconstruction_loss_batch * window_length * number_of_columns
    
    def total_loss(y_true, y_pred, distribution_mean, distribution_variance):
        reconstruction_loss_batch = get_reconstruction_loss(y_true, y_pred)
        kl_loss_batch = kl_loss_layer([distribution_mean, distribution_variance])
        return reconstruction_loss_batch + kl_loss_batch
        
    return total_loss

# Instantiate the CVAE model
cvae = CVAE(latent_features)

# Define the weights file with an extension that ends with ".weights.h5"
weights_file = os.path.join(model_save_path, 'model_weights.weights.h5')

# Load saved weights if available
if use_saved_weights and os.path.exists(weights_file):
    print("Loading saved weights...")
    cvae.load_weights(weights_file)
else:
    print("Using random initialization...")

# Prepare optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)

# Prepare the loss function using the model's KL loss layer
loss_fn = get_loss(cvae.kl_loss_layer)

# Define the training step
@tf.function
def train_step(model, x, optimizer, loss_fn):
    """Executes one training step and returns the loss."""
    with tf.GradientTape() as tape:
        mean, logvar = model.encode(x)
        z = model.reparameterize(mean, logvar)
        x_logit = model.decode(z)
        loss = loss_fn(x, x_logit, mean, logvar)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# Training loop (only run if you plan to train a new model)
if not use_saved_weights or not os.path.exists(weights_file):
    for epoch in range(1, epochs + 1):
        for batch in range(0, len(X_train), batch_size):
            train_x = X_train[batch:batch + batch_size]
            loss = train_step(cvae, train_x, optimizer, loss_fn)
        print(f"Epoch {epoch}, Loss: {loss.numpy()}")
    
    # Save only the model weights after training
    cvae.save_weights(weights_file)
    print("Model weights saved.")

#%%

# def plot_reconstruction(data, model, case=170):

#     # After filtering, the data should already have 12 columns.
#     # Create joint names based on the selected indices.
#     selected_joint_names = [columns_names[i] for i in selected_indices]
#     num_joints = len(selected_joint_names)
    
#     # Create a 12 x 2 subplot grid
#     fig, axes = plt.subplots(num_joints, 2, figsize=(15, num_joints * 2))
    
#     # Define the x-axis ticks for a 200-timestep sequence with 6 ticks (can be percentages)
#     x_ticks = np.linspace(0, 200, 6)
#     x_tick_labels = ['0', '20', '40', '60', '80', '100']
    
#     # Plot original data for each joint angle (directly index the already-filtered channels)
#     for j in range(num_joints):
#         axes[j, 0].plot(data[case, :200, j], color='blue')
#         axes[j, 0].set_title(f'{selected_joint_names[j]} (Original)')
#         axes[j, 0].set_xticks(x_ticks)
#         axes[j, 0].set_xticklabels(x_tick_labels)
#         axes[j, 0].set_ylabel("Angle Value")
    
#     # Reconstruct the data for the given case once
#     visualDecodedData = np.expand_dims(data[case], axis=0)
#     mean, logvar = model.encode(visualDecodedData)
#     latent_encoding = model.reparameterize(mean, logvar)
#     reconstructed_data = model.decode(latent_encoding).numpy()

#     # Plot reconstructed data for each joint angle
#     for j in range(num_joints):
#         axes[j, 1].plot(reconstructed_data[0, :200, j], color='red')
#         axes[j, 1].set_title(f'{selected_joint_names[j]} (Reconstructed)')
#         axes[j, 1].set_xticks(x_ticks)
#         axes[j, 1].set_xticklabels(x_tick_labels)
#         axes[j, 1].set_ylabel("Angle Value")
    
#     plt.tight_layout()
#     plt.show()

# # Example usage:
# plot_reconstruction(X_test, cvae, case=170)

#%%

# Extract latent features for IMU data using your per-sample approach
encoded_IMU = []
for i in range(len(data_IMU)):
    sample = np.expand_dims(data_IMU[i], axis=0)
    mean, logvar = cvae.encode(sample)
    latent_encoding = cvae.reparameterize(mean, logvar)
    # latent_encoding[0] extracts the vector from the batch dimension
    encoded_IMU.append(latent_encoding[0].numpy())

encoded_IMU = np.array(encoded_IMU)

# Extract latent features for OMC data using the same approach
encoded_OMC = []
for i in range(len(data_OMC)):
    sample = np.expand_dims(data_OMC[i], axis=0)
    mean, logvar = cvae.encode(sample)
    latent_encoding = cvae.reparameterize(mean, logvar)
    encoded_OMC.append(latent_encoding[0].numpy())

encoded_OMC = np.array(encoded_OMC)

# 3D Plot: Plot all three latent variables for IMU and OMC data
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot IMU latent features in blue
ax.scatter(encoded_IMU[:, 0], encoded_IMU[:, 1], encoded_IMU[:, 2],
            c='blue', label='IMU Data', alpha=0.6)

# Plot OMC latent features in red
ax.scatter(encoded_OMC[:, 0], encoded_OMC[:, 1], encoded_OMC[:, 2],
            c='red', label='OMC Data', alpha=0.6)

# Set axis labels and title
ax.set_xlabel('Latent Variable 1')
ax.set_ylabel('Latent Variable 2')
ax.set_zlabel('Latent Variable 3')
ax.set_title('3D Latent Features from IMU and OMC Data')
ax.legend()

plt.show()

# #%%

# import matplotlib.lines as mlines

# # Define color maps for the two modalities (for participant S08 only)
# imu_color_map = {
#     "walk35": "lightblue",
#     "walk45": "blue",
#     "walk55": "darkblue"
# }
# omc_color_map = {
#     "walk35": "lightcoral",
#     "walk45": "red",
#     "walk55": "darkred"
# }

# # ========================
# # Process IMU Data
# # ========================
# encoded_IMU = []
# colors_IMU = []

# for i in range(len(data_IMU)):
#     sample = np.expand_dims(data_IMU[i], axis=0)
#     mean, logvar = cvae.encode(sample)
#     latent_encoding = cvae.reparameterize(mean, logvar)
#     encoded_IMU.append(latent_encoding[0].numpy())
    
#     participant_id = ids_IMU_OMC[i, 0]  # e.g. "S08"
#     walk_speed = ids_IMU_OMC[i, 1]      # e.g. "walk35"
    
#     # For IMU: If participant is S08, assign a blue shade by walk speed; otherwise, grey.
#     if participant_id == "S08":
#         colors_IMU.append(imu_color_map.get(walk_speed, "blue"))
#     else:
#         colors_IMU.append("grey")

# encoded_IMU = np.array(encoded_IMU)
# colors_IMU = np.array(colors_IMU)

# # ========================
# # Process OMC Data
# # ========================
# encoded_OMC = []
# colors_OMC = []

# for i in range(len(data_OMC)):
#     sample = np.expand_dims(data_OMC[i], axis=0)
#     mean, logvar = cvae.encode(sample)
#     latent_encoding = cvae.reparameterize(mean, logvar)
#     encoded_OMC.append(latent_encoding[0].numpy())
    
#     participant_id = ids_IMU_OMC[i, 0]  # e.g. "S08"
#     walk_speed = ids_IMU_OMC[i, 1]
    
#     # For OMC: If participant is S08, assign a red shade by walk speed; otherwise, grey.
#     if participant_id == "S08":
#         colors_OMC.append(omc_color_map.get(walk_speed, "red"))
#     else:
#         colors_OMC.append("grey")

# encoded_OMC = np.array(encoded_OMC)
# colors_OMC = np.array(colors_OMC)

# # ========================
# # Separate indices for grey vs. participant S08 in each modality
# # For IMU
# imu_grey_idx = np.where(colors_IMU == "grey")[0]
# imu_s08_idx = np.where(colors_IMU != "grey")[0]

# # For OMC
# omc_grey_idx = np.where(colors_OMC == "grey")[0]
# omc_s08_idx = np.where(colors_OMC != "grey")[0]
# #%%

# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')

# # Plot grey points for IMU (more transparent)
# ax.scatter(encoded_IMU[imu_grey_idx, 0], encoded_IMU[imu_grey_idx, 1], encoded_IMU[imu_grey_idx, 2],
#            c="grey", marker='o', alpha=0.02, label='IMU (Other Participants)')

# # Plot S08 points for IMU (using blue shades) with full opacity
# ax.scatter(encoded_IMU[imu_s08_idx, 0], encoded_IMU[imu_s08_idx, 1], encoded_IMU[imu_s08_idx, 2],
#            c=colors_IMU[imu_s08_idx], marker='o', alpha=0.7, label='IMU S08')

# # Plot grey points for OMC (more transparent)
# ax.scatter(encoded_OMC[omc_grey_idx, 0], encoded_OMC[omc_grey_idx, 1], encoded_OMC[omc_grey_idx, 2],
#            c="grey", marker='^', alpha=0.02, label='OMC (Other Participants)')

# # Plot S08 points for OMC (using red shades) with full opacity
# ax.scatter(encoded_OMC[omc_s08_idx, 0], encoded_OMC[omc_s08_idx, 1], encoded_OMC[omc_s08_idx, 2],
#            c=colors_OMC[omc_s08_idx], marker='^', alpha=0.7, label='OMC S08')

# ax.set_xlabel("Latent Variable 1")
# ax.set_ylabel("Latent Variable 2")
# ax.set_zlabel("Latent Variable 3")
# ax.set_title("3D Latent Space Representation: IMU vs. OMC Data (Grey Transparency Changed)")

# # ========================
# # Custom Legend
# # ========================
# legend_handles = []

# # Grey for Other Participants (common entry for both modalities)
# legend_handles.append(mlines.Line2D([], [], color='grey', marker='o', linestyle='None', markersize=8, label='Other Participants'))

# # Legend for S08, IMU (circles) with different walk speeds
# for ws, col in imu_color_map.items():
#     legend_handles.append(mlines.Line2D([], [], color=col, marker='o', linestyle='None',
#                                           markersize=8, label=f'S08 - {ws} (IMU)'))

# # Legend for S08, OMC (triangles) with different walk speeds
# for ws, col in omc_color_map.items():
#     legend_handles.append(mlines.Line2D([], [], color=col, marker='^', linestyle='None',
#                                           markersize=8, label=f'S08 - {ws} (OMC)'))

# ax.legend(handles=legend_handles, loc='best')

# plt.show()
