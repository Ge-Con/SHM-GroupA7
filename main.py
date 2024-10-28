# Import external libraries
import pandas as pd
import numpy as np
import os

# Import modules
import extract_features
import PCA
from Signal_Processing import Data_Preprocess
from prognosticcriteria_v2 import fitness, test_fitness
from DeepSAD_reduced import DeepSAD_train_run, plot_ds_images
import Graphs
import SP_save as SP
from Interpolating import scale_exact
from Data_concatenation import process_csv_files
from OLD_VAE import Hyper as HYP
import Hyper_save_parameters as HYPparameters
import Hyper_save_HI as HYPresults
from matplotlib import pyplot as plt
import tensorflow as tf
import csv

# Set options
pd.set_option('display.max_columns', 15)
pd.set_option('display.width', 400)
# Set options for numpy
np.set_printoptions(linewidth=400)
np.set_printoptions(precision=4, suppress=True)


def saveFeatures(dir):
    """
    Feature Extraction: Processes files with names ending in 'kHz.csv', 'FFT.csv', 'HLB.csv',
    'EMD.csv' and 'SFT.csv' and saves the features in a new .csv file with '_FT' appended before the extension.

    Parameters:
        dir (str): The directory path containing CSV files

    Returns:
        None
    """

    print("Extracting Features:...")
    for root, dirs, files in os.walk(dir):
        for name in files:

            # Time domain extraction for 'kHz.csv' files
            if name.endswith('kHz.csv'):
                data = pd.read_csv(os.path.join(root, name))
                features = extract_features.time_to_feature(data)
                new_filename = name.replace('kHz.csv', 'kHz_FT.csv')
                csv_file_path = os.path.join(root, new_filename)
                features.to_csv(csv_file_path, index=False)

            # FFT feature extraction
            elif name.endswith('FFT.csv'):
                data = pd.read_csv(os.path.join(root, name))
                features = extract_features.freq_to_feature(data)
                new_filename = name.replace('FFT.csv', 'FFT_FT.csv')
                csv_file_path = os.path.join(root, new_filename)
                features.to_csv(csv_file_path, index=False)

            # Hilbert transform (HLB) feature extraction
            elif name.endswith('HLB.csv'):
                data = pd.read_csv(os.path.join(root, name))
                features = extract_features.time_to_feature(data)
                new_filename = name.replace('HLB.csv', 'HLB_FT.csv')
                csv_file_path = os.path.join(root, new_filename)
                features.to_csv(csv_file_path, index=False)

            # Empirical Mode Decomposition (EMD) feature extraction
            elif name.endswith('EMD.csv'):
                data = pd.read_csv(os.path.join(root, name))
                features = extract_features.time_to_feature(data)
                new_filename = name.replace('EMD.csv', 'EMD_FT.csv')
                csv_file_path = os.path.join(root, new_filename)
                features.to_csv(csv_file_path, index=False)

            # Short-Time Fourier Transform (STFT) feature extraction
            elif name.endswith('SFT.csv'):
                data = pd.read_csv(os.path.join(root, name))
                # Unflatten to 3D array (assuming data structure is known)
                data3d = [[[0 for _ in range(17)] for _ in range(126)] for _ in range(56)]
                for k in range(56):
                    for i in range(126):
                        for j in range(17):
                            data3d[k][i][j] = data.iloc[126*k + i, j]

                features = extract_features.STFT_to_feature(data3d)
                new_filename = name.replace('SFT.csv', 'SFT_FT.csv')
                csv_file_path = os.path.join(root, new_filename)
                features.to_csv(csv_file_path, index=False)

def FFT_HLB_Reduced_Feat(dir):
    """
    Save reduced features for FFT and Hilbert transforms

    Parameters:
        dir (str): The directory path containing CSV files

    Returns:
        None
    """

    print("Extracting Features:...")
    for root, dirs, files in os.walk(dir):
        for name in files:

            # FFT feature reduction
            if name.endswith('FFT.csv'):
                data = pd.read_csv(os.path.join(root, name))
                features = extract_features.FFT_Feat_reduced(data)
                new_filename = name.replace('FFT.csv', 'FFT_FT_Reduced.csv')
                csv_file_path = os.path.join(root, new_filename)
                features.to_csv(csv_file_path, index=False)

            # Hilbert (HLB) feature reduction
            elif name.endswith('HLB.csv'):
                data = pd.read_csv(os.path.join(root, name))
                features = extract_features.HLB_Feat_reduced(data)
                new_filename = name.replace('HLB.csv', 'HLB_FT_Reduced.csv')
                csv_file_path = os.path.join(root, new_filename)
                features.to_csv(csv_file_path, index=False)


def AverageFeatures(rootdir):
    """
    Averages and saves extracted features across paths.

    Parameters:
        rootdir (str): The directory path containing CSV files

    Returns:
        None
    """

    frequencies = ["050", "100", "125", "150", "200", "250"]
    samples = ["PZT-CSV-L1-03", "PZT-CSV-L1-04", "PZT-CSV-L1-05", "PZT-CSV-L1-09", "PZT-CSV-L1-23"]

    print("Combining Features:...")
    for sample in samples:
        dir = os.path.join(rootdir, sample)

        # Create All Features (AF) files & Check if file matches the frequency and file type
        for root, dirs, files in os.walk(dir):
            all_feat = np.empty((6, 5), dtype=object)
            flag = False

            for name in files:
                for freq in frequencies:
                    if freq in name and name.endswith('_FT.csv'):
                        flag = True
                        data = np.array(pd.read_csv(os.path.join(root, name)))
                        if 'FFT' in name:
                            all_feat[frequencies.index(freq)][1] = data
                        elif 'HLB' in name:
                            all_feat[frequencies.index(freq)][2] = data
                        elif 'EMD' in name:
                            all_feat[frequencies.index(freq)][3] = data
                        elif 'SFT' in name:
                            all_feat[frequencies.index(freq)][4] = data
                        else:  # Time domain
                            all_feat[frequencies.index(freq)][0] = data

            # Save found features to separate folder as AF. Flag = at least one file was the correct type
            if flag:
                for freq in ["050", "100", "125", "150", "200", "250"]:
                    combinedfeatures = np.concatenate(
                        [all_feat[frequencies.index(freq), i] for i in range(all_feat[0].shape[0])], axis=0)
                    root_new = root.replace("PZT", "PZT-ONLY-FEATURES")
                    if not os.path.exists(root_new):
                        os.makedirs(root_new)
                    csv_file_path = os.path.join(root_new, freq + "kHz_AF.csv")
                    pd.DataFrame(combinedfeatures).to_csv(csv_file_path, index=False)

        # Average features at each state
        print("Averaging features...")
        meanfeatures = np.empty((6), dtype=object)
        meanfeatures[:] = None

        # Read All Features files and calculate Mean Features
        for root, dirs, files in os.walk(dir):
            for name in files:
                if name.endswith("kHz_AF.csv"):
                    data = np.array(pd.read_csv(os.path.join(root, name)))
                    data_mean = np.mean(data, axis=0)
                    freq_index = frequencies.index(name[:3])
                    if meanfeatures[freq_index] is None:
                        meanfeatures[freq_index] = np.array([data_mean])
                    else:
                        meanfeatures[freq_index] = np.concatenate((meanfeatures[freq_index], np.array([data_mean])),
                                                                  axis=0)
        # Save Mean Features
        for freq in frequencies:  # For each feature
            freq_index = frequencies.index(freq)
            if meanfeatures[freq_index] is not None:
                csv_file_path = os.path.join(dir, freq + "kHz_MF.csv")  # MF = Mean Features
                pd.DataFrame(meanfeatures[freq_index]).to_csv(csv_file_path, index=False)
'''
#
# def correlateFeatures(rootdir):
#     """
#         Correlates extracted average features, saving correlation matrix and reduced feature matrices.
#         This is no longer used in the framework, but is kept for completeness.
#
#         Parameters:
#             rootdir(str): The directory path containing CSV files
#         Returns: None
#     """
#
#     frequencies = ["050", "100", "125", "150", "200", "250"]
#     samples = ["PZT-CSV-L1-03", "PZT-CSV-L1-04", "PZT-CSV-L1-05", "PZT-CSV-L1-09", "PZT-CSV-L1-23"]
#
#     # Walk folders of each sample in turn
#     for sample in samples:
#         dir = rootdir + "\\" + sample
#
#         # Read All Features for each state and calculate Mean Features
#         print("Reading Files: " + sample)
#         meanfeatures = np.empty((6), dtype=object)
#         for root, dirs, files in os.walk(dir):
#             for name in files:
#                 if name.endswith("kHz_AF.csv"):
#                     data = np.array(pd.read_csv(os.path.join(root, name)))
#                     data = np.mean(data, axis=1)
#                     if str(type(meanfeatures[frequencies.index(name[:3])])) == "<class 'NoneType'>":    # First to be added
#                         meanfeatures[frequencies.index(name[:3])] = np.array([data])
#                     else:   # Concatenate subsequent entries
#                         meanfeatures[frequencies.index(name[:3])] = np.concatenate((meanfeatures[frequencies.index(name[:3])], np.array([data])))
#
#         # Correlate mean features
#         print("Correlating Features:...")
#         alldelete = np.empty((6), dtype=object)
#         for freq in frequencies:
#             csv_file_path = os.path.join(dir, freq + "kHz_MF.csv") # MF = Mean Features
#             # Read mean features from file
#             pd.DataFrame(meanfeatures[frequencies.index(freq)]).to_csv(csv_file_path, index=False)
#
#             #Use feature_correlation to return the correlation matrix, reduced matrix and features deleted
#             correlation_matrix, features, to_delete = extract_features.feature_correlation(meanfeatures[frequencies.index(freq)])
#
#             #Save all to files
#             alldelete[frequencies.index(freq)] = to_delete
#             csv_file_path = os.path.join(dir, freq + "kHz_CF.csv") # CM = Correlated feature matrix
#             pd.DataFrame(correlation_matrix).to_csv(csv_file_path, index=False)
#             csv_file_path = os.path.join(dir, freq + "kHz_RF.csv") # RF = Remainder features
#             pd.DataFrame(features).to_csv(csv_file_path, index=False)
#
#         # Save lists of deleted features for inspection
#         csv_file_path = os.path.join(dir, "DF.csv") # DF = Deleted features
#         pd.DataFrame(alldelete).to_csv(csv_file_path, index=False)
#
'''
def savePCA(dir): #Calculates and saves 1 principle component PCA
    """
        Execute and save PCA

        Parameters:
            dir(str): The directory path containing CSV files
        Returns: None
    """

    # Number of principal components to be calculated - change as desired
    pcs_upto = 3

    # File types for input
    filenames = ["FFT_FT_Reduced", "HLB_FT_Reduced"]            # No .csv
    output = np.zeros((6, pcs_upto * len(filenames), 5, 5, 30)) # 6 frequencies, n PCs, 5 folds, 5 test panels, 30 states

    # Execute PCA on each file and on each PC
    for file in range(len(filenames)):
        for pc in range(pcs_upto):
            # Temporarily store the result before saving to correct locations in output array
            tempout = PCA.doPCA_multiple_Campaigns(dir, filenames[file], pc+1)
            for freq in range(6):
                output[freq][pc+file*pcs_upto] = tempout[freq]

    # Save and graph HIs
    labels = np.array(["Sample 1", "Sample 2", "Sample 3", "Sample 4", "Sample 5"])
    save_evaluation(np.array(output), "PCA", dir, labels)


def save_evaluation(features, label, dir, files_used=[""]):
    """
    Save basic CSVs and graphs of prognostic criteria applied to Health Indices (HIs).

    Parameters:
        features (3D np array): Extracted feature data or HI output for all samples. Can be 2D or 3D.
        label (str): The label used for saving results.
        dir (str): Directory to save the results.
        files_used (list): List of feature names used, if any (optional).

       Returns:
        criteria (3D np array): Results of prognostic criteria with 4 types of criteria, 6 frequencies, and n features.
    """

    # Create unique save directory for HIs
    count = 1
    # savedir = dir + '\\' + label
    savedir = os.path.join(dir, label)
    while os.path.exists(savedir + '.npy'):
        # savedir = dir + '\\' + label + str(count)
        savedir = os.path.join(dir, label + str(count))
        count += 1

    # Save HIs to .npy file for use by weighted average ensemble (WAE)
    np.save(savedir, features)

    # Recursive call if the features array is multi-dimensional (more than 4D)
    if features.ndim > 4:
        criteria = []
        for i in range(features.shape[1]):
            criteria.append(save_evaluation(features[:, i], label + " framework " + str(i+1), dir, files_used))
        return criteria

    #Else(features.ndim < 4)
    frequencies = ["050", "100", "125", "150", "200", "250"]

    # Initialize the criteria array to store fitness, monotonicity, trendability, and prognosability
    criteria = np.empty((4, 6, len(features[0])))
    featuresonly = files_used[0] == ""

    # Evaluate prognostic criteria for each frequency and feature
    for freq in range(6):
        print("Evaluating: " + frequencies[freq] + "kHz")
        for feat in range(len(features[0])):
            if feat % 50 == 0 and feat != 0:    # Print occasional progress updates
                print(feat, " features")

            # Evaluate and store prognostic criteria and store the results
            features[freq][feat] = np.array(features[freq][feat])
            ftn, mo, tr, pr, error = fitness(features[freq][feat])
            criteria[0][freq][feat] = float(ftn)
            criteria[1][freq][feat] = float(mo)
            criteria[2][freq][feat] = float(pr)
            criteria[3][freq][feat] = float(tr)

    # Save criteria to CSV files
    print("Saving to CSVs")
    pd.DataFrame(criteria[0]).to_csv(os.path.join(dir, label + " Fit.csv"), index=False)  # Feature against frequency
    pd.DataFrame(criteria[1]).to_csv(os.path.join(dir, label + " Mon.csv"), index=False)
    pd.DataFrame(criteria[2]).to_csv(os.path.join(dir, label + " Tre.csv"), index=False)
    pd.DataFrame(criteria[3]).to_csv(os.path.join(dir, label + " Pro.csv"), index=False)

    # Generate graphs of Health Indices (HIs)
    for freq in range(6):
        print("Graphing: " + frequencies[freq] + "kHz")
        for feat in range(len(features[0])):
            Graphs.HI_graph(features[freq][feat], dir=dir, name=f"{label}-{frequencies[freq]}-{feat}")

        # Generate bar charts of prognostic criteria against features
        if featuresonly:
            files_used = np.array([str(i) for i in range(len(features[0]))])
        try:
            Graphs.criteria_chart(files_used, criteria[1][freq], criteria[2][freq], criteria[3][freq], dir=dir, name=f"{label}-{frequencies[freq]}")
        # In case the wrong number of labels was given
        except ValueError:
            print("ValueError, bar chart not generated")

    # Simple average of prognostic criteria across frequencies
    if featuresonly:
        avs = np.empty((4, 2), dtype=object)
        for crit in range(4):
            avs[crit, 0] = np.expand_dims(np.mean(criteria[crit], axis= 0),axis=0)[0]
            avs[crit, 1] = np.std(criteria[crit], axis = 0)
        Graphs.criteria_chart(files_used, avs[1][0], avs[2][0], avs[3][0], dir=dir, name=f"{label}-Av")
        av_arr = np.vstack((avs[0, 0], avs[0, 1]))
        pd.DataFrame(av_arr).to_csv(os.path.join(dir, label + " Fit AF.csv"), index=False)

        return criteria

def retrieve_features(dir):
    """
    Read feature data from CSV files and return it as a numpy array.

    Parameters:
        dir (str): Root directory containing the CSV files.

    Returns:
        features (4D np array): Signal features array of shape [6 frequencies x 139 features x 5 samples x n states].
    """

    frequencies = ["050", "100", "125", "150", "200", "250"]
    features = np.empty((6, 139), dtype=object)

    # Walk through the directory to find and process CSV files
    for root, dirs, files in os.walk(dir):
        for name in files:
            if name.endswith("MF.csv"):
                # Read data from the CSV file and determine the frequency from the filename
                data = np.array(pd.read_csv(os.path.join(root, name))).transpose()
                freq = frequencies.index(name[:3])

                # Add each feature to correct position in the features array
                for feat in range(139):
                    if str(type(features[freq][feat])) == "<class 'NoneType'>":
                        features[freq][feat] = np.array([scale_exact(data[feat])])
                    else:
                        features[freq][feat] = np.vstack([features[freq][feat], scale_exact(data[feat])])

    return features


def hyperDeepSad(dir):
    """
        Optimise hyperparameters for DeepSAD

        Parameters:
        - dir (str): CSV root folder directory
        Returns: None
    """

    # List frequencies, filenames and samples
    frequencies = ["050", "100", "125", "150", "200", "250"]
    filenames = ["FFT_FT_Reduced", "HLB_FT_Reduced"]
    samples = ["PZT-FFT-HLB-L1-03", "PZT-FFT-HLB-L1-04", "PZT-FFT-HLB-L1-05", "PZT-FFT-HLB-L1-09", "PZT-FFT-HLB-L1-23"]

    # Optimise hyperparameters for all files and frequencies
    for file in filenames:
        for freq in frequencies:
            params = DeepSAD_train_run(dir, freq, file, True)

            # Save to external file
            for sample in range(5):
                HYP.simple_store_hyperparameters(params[sample], file, samples[sample], freq, dir)


def saveDeepSAD(dir):
    """
        Generate and save DeepSAD HIs

        Parameters:
        - dir (str): CSV root folder directory
        Returns: None
    """

    # List frequencies and filenames
    frequencies = ["050", "100", "125", "150", "200", "250"]
    types = ["FFT", "HLB"]
    filename = "_FT_Reduced"
    # filename_FFT = "FFT_FT_Reduced"
    # filename_HLB = "HLB_FT_Reduced"

    # Initialise HI arrays
    HIs_FFT = np.empty((6), dtype=object)
    HIs_HLB = np.empty((6), dtype=object)

    # Generate HIs for all frequencies from FFT features
    for freq in range(len(frequencies)):
        print(f"Processing frequency: {frequencies[freq]} kHz for FFT")
        HIs_FFT[freq] = DeepSAD_train_run(dir, frequencies[freq], "FFT" + filename)

    # Save and plot results
    save_evaluation(np.array(HIs_FFT), "DeepSAD_FFT", dir, filename)
    plot_ds_images(dir, "FFT")

    # Generate HIs for all frequencies from Hilbert features
    for freq in range(len(frequencies)):
        print(f"Processing frequency: {frequencies[freq]} kHz for HLB")
        HIs_HLB[freq] = DeepSAD_train_run(dir, frequencies[freq], "HLB" + filename)

    # Save and plot results
    save_evaluation(np.array(HIs_HLB), "DeepSAD_HLB", dir, filename)
    plot_ds_images(dir, "HLB")


def hyperVAE(dir, concatenate=False):
    """
        Optimise hyperparameters for VAE

        Parameters:
        - dir (str): CSV root folder directory
        Returns: None

        This function is a work in progress, VAE is currently run independently
    """
    # Connect global variable for seed
    global vae_seed

    # Set random seeds
    tf.compat.v1.reset_default_graph()
    tf.random.set_seed(vae_seed)
    np.random.seed(vae_seed)

    # List frequencies, filenames and samples
    freqs = ("050_kHz", "100_kHz", "125_kHz", "150_kHz", "200_kHz", "250_kHz")
    filenames = ["FFT_FT_Reduced", "HLB_FT_Reduced"]
    samples = ["PZT-FFT-HLB-L1-03", "PZT-FFT-HLB-L1-04", "PZT-FFT-HLB-L1-05", "PZT-FFT-HLB-L1-09", "PZT-FFT-HLB-L1-23"]

    #TODO: why is this different to samples
    panels = ("L103", "L105", "L109", "L104", "L123")

    # Concatenate input files
    if concatenate:
        for panel in samples:
            for file in filenames:
                process_csv_files(dir, panel, file)

    # Optimise hyperparameters for VAE
    for file_type in filenames:
        counter = 0
        for panel in panels:
            for freq in freqs:

                # Create list of file paths
                train_filenames = []
                for i in tuple(x for x in panels if x != panel):
                    filename = os.path.join(dir, f"concatenated_{freq}_{i}_{file_type}.csv")
                    train_filenames.append(filename)

                #Output progress
                counter += 1
                print("Counter: ", counter)
                print("Panel: ", panel)
                print("Freq: ", freq)
                print("SP Features: ", file_type)

                #TODO: This is concerning, especially because it's commented here but not for the test data
                # For va                vae_train_data.drop(vae_train_data.columns[len(vae_train_data.columns) - 1], axis=1, inplace=True)e_train_data, create the merged file with all 4 panels, delete the last column. I don't know why we delete the last column though
                vae_train_data, flags = HYPparameters.mergedata(train_filenames)

                # Read test data
                test_filename = os.path.join(dir, f"concatenated_{freq}_{panel}_{file_type}.csv")
                vae_test_data = pd.read_csv(test_filename, header=None).values.transpose()
                vae_test_data = np.delete(vae_test_data, -1, axis=1)

                # Normalize the train and test data, with respect to the train data
                vae_scaler = HYPparameters.StandardScaler()
                vae_scaler.fit(vae_train_data)
                vae_train_data = vae_scaler.transform(vae_train_data)
                vae_test_data = vae_scaler.transform(vae_test_data)

                # Apply PCA to the train and test data, fit to the train data
                vae_pca = HYPparameters.PCA(n_components=30)
                vae_pca.fit(vae_train_data)
                vae_train_data = vae_pca.transform(vae_train_data)
                vae_test_data = vae_pca.transform(vae_test_data)

                # Perform hyperparameter optimisation and save todo how are they saved
                hyperparameters = HYPparameters.hyperparameter_optimisation(vae_train_data, vae_test_data, vae_scaler, vae_pca,
                                                              vae_seed, file_type, panel, freq, dir, n_calls=40)
                HYPparameters.simple_store_hyperparameters(hyperparameters, file_type, panel, freq, dir)

def saveVAE(dir, save_graph=True, save_HI=True, valid=False):   #todo: remove validation if not working, then get rid of the options as well
    """
        Run and save VAE HIs
    
        Parameters:
        - dir (str): CSV root folder directory
        Returns: None

        This function is a work in progress, VAE is currently run independently
    """

    # Connect global variable for seed
    global vae_seed

    # Set random seeds
    tf.compat.v1.reset_default_graph()
    tf.random.set_seed(vae_seed)
    np.random.seed(vae_seed)

    # List frequencies, filenames and samples
    freqs = ("050_kHz", "100_kHz", "125_kHz", "150_kHz", "200_kHz", "250_kHz")
    filenames = ["FFT_FT_Reduced", "HLB_FT_Reduced"]
    panels = ("L103", "L105", "L109", "L104", "L123")

    # Determine dimensions of data
    time_steps = 30
    num_HIs = 5
    num_freqs = len(freqs)
    num_panels = len(panels)

    #todo bahaha no I think Pablo needs to do this
    for file_type in filenames:
        counter = 0
        result_dictionary = {}
        hyperparameters_df = pd.read_csv(os.path.join(dir, f'hyperparameters-opt-{file_type}.csv'), index_col=0)
        hi_full_array = np.zeros((num_panels, num_freqs, num_HIs, time_steps))

        for panel_idx, panel in enumerate(panels):
            for freq_idx, freq in enumerate(freqs):

                train_filenames = []

                for i in tuple(x for x in panels if x != panel):
                    filename = os.path.join(dir, f"concatenated_{freq}_{i}_{file_type}.csv")
                    train_filenames.append(filename)

                if valid:
                    valid_panel = panels[panel_idx-1]
                    train_filenames.pop(valid_panel)

                result_dictionary[f"{panel}{freq}"] = []

                counter += 1
                print("Counter: ", counter)
                print("Panel: ", panel)
                print("Freq: ", freq)
                print("SP Features: ", file_type)

                vae_train_data, flags = HYPparameters.mergedata(train_filenames)
                vae_train_data.drop(vae_train_data.columns[len(vae_train_data.columns) - 1], axis=1, inplace=True)

                test_filename = os.path.join(dir, f"concatenated_{freq}_{panel}_{file_type}.csv")
                vae_test_data = pd.read_csv(test_filename, header=None).values.transpose()
                vae_test_data = np.delete(vae_test_data, -1, axis=1)

                if valid:
                    valid_filename = os.path.join(dir, f"concatenated_{freq}_{valid_panel}_{file_type}.csv")
                    vae_valid_data = pd.read_csv(valid_filename, header=None).values.transpose()
                    vae_valid_data = np.delete(vae_valid_data, -1, axis=1)

                vae_scaler = HYPparameters.StandardScaler()
                vae_scaler.fit(vae_train_data)
                vae_train_data = vae_scaler.transform(vae_train_data)
                vae_test_data = vae_scaler.transform(vae_test_data)
                if valid:
                    vae_valid_data = vae_scaler.transform(vae_valid_data)

                vae_pca = HYPparameters.PCA(n_components=30)
                vae_pca.fit(vae_train_data)
                vae_train_data = vae_pca.transform(vae_train_data)
                vae_test_data = vae_pca.transform(vae_test_data)
                if valid:
                    vae_valid_data = vae_pca.transform(vae_valid_data)

                hyperparameters_str = hyperparameters_df.loc[freq, panel]
                hyperparameters = eval(hyperparameters_str)

                if valid:
                    health_indicators = HYPparameters.train_vae(hyperparameters[0][0], hyperparameters[0][1],
                                              hyperparameters[0][2], hyperparameters[0][3], hyperparameters[0][4], hyperparameters[0][5], hyperparameters[0][6],
                                                            vae_train_data, vae_test_data, vae_scaler, vae_pca, vae_seed, file_type, panel, freq, dir, valid=valid, valid_data=vae_valid_data)
                else:
                    health_indicators = HYPparameters.train_vae(hyperparameters[0][0], hyperparameters[0][1],
                                              hyperparameters[0][2], hyperparameters[0][3], hyperparameters[0][4], hyperparameters[0][5], hyperparameters[0][6],
                                                            vae_train_data, vae_test_data, vae_scaler, vae_pca, vae_seed, file_type, panel, freq, dir, valid=valid, valid_data=None)

                fitness_all = fitness(health_indicators[0])
                print("Fitness all", fitness_all)

                fitness_test = test_fitness(health_indicators[2], health_indicators[1])
                print("Fitness test", fitness_test)

                if valid:
                    fitness_valid = test_fitness(health_indicators[6], health_indicators[1])
                    print("Fitness valid", fitness_valid)
                    result_dictionary[f"{panel}{freq}"].append([fitness_all, fitness_test, fitness_valid])

                else:
                    result_dictionary[f"{panel}{freq}"].append([fitness_all, fitness_test])

                if save_graph:
                    graph_hi_filename = f"HI_graph_{freq}_{panel}_{file_type}_seed_{vae_seed}"
                    graph_hi_dir = os.path.join(dir, graph_hi_filename)

                    fig = plt.figure()

                    train_panels = [k for k in panels if k != panel]

                    x = np.arange(0, health_indicators[2].shape[1], 1)
                    x = x * (1 / (x.shape[0] - 1))
                    x = x * 100

                    for i in range(len(train_panels)):
                        hi = health_indicators[1][i]
                        std_dev = health_indicators[4][i]

                        plt.errorbar(x, hi, yerr=std_dev, label=f'Sample {i + 1}: Train', color=f'C{i}', ecolor='blue', elinewidth=2, capsize=5)

                    plt.errorbar(x, health_indicators[2][0], yerr=health_indicators[5][0], label=f'Sample {panels.index(panel) + 1}: Test', color='red', ecolor='salmon', elinewidth=2, capsize=5)
                    if valid:
                        plt.errorbar(x, health_indicators[6][0], yerr=health_indicators[7][0], label=f'Sample {panels.index(valid_panel) + 1}: Validation', color='purple', ecolor='cyan', elinewidth=2, capsize=5)

                    plt.xlabel('Lifetime (%)')
                    plt.ylabel('Health Indicators')
                    plt.title('Train and Test Health Indicators over Time')
                    plt.legend()

                    plt.savefig(graph_hi_dir)
                    plt.close(fig)

                fitness_test = (fitness_test)
                fitness_all = (fitness_all)
                if valid:
                    fitness_valid = (fitness_valid)
                HYPresults.store_hyperparameters(fitness_all, fitness_test, panel, freq, file_type, vae_seed, dir)

                z_all_modified = np.array([scale_exact(row) for row in health_indicators[0][:num_HIs]])
                z_all_modified = z_all_modified.reshape(num_HIs, time_steps)

                # Assign z_all_modified to the correct position in hi_full_array
                hi_full_array[panel_idx, freq_idx] = z_all_modified

        if save_HI:
            label = f"VAE_{file_type}_seed_{vae_seed}"
            savedir = dir + '\\' + label
            np.save(savedir, hi_full_array)

        if save_graph:
            HYPresults.plot_images(vae_seed, file_type, dir)

        with open(f"results_{file_type}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, result_dictionary.keys())
            w.writeheader()
            w.writerow(result_dictionary)

def main_menu():
    """
        Print options in main menu

        Parameters: None
        Returns: None
    """

    # Output options for menu
    print("\n---")
    print("Welcome! Please choose a signal processing method: ")
    print("0. Exit")
    print("1. Extract PZTs")
    print("")
    print("From raw CSVs:")
    print("2. Carry out SP transforms")
    print("3. Extract and average all features")
    print("4. Extract reduced features and export FFT and Hilbert only (on a separate directory)")
    print("5. Evaluate all feature HIs (Requires MF.csv in directory)")
    print("")
    print("From FFT & Hilbert, raw & features:")
    print("6. Execute and evaluate PCA")
    print("7. Train VAE hyperparameters (and concatenate data if necessary)")
    print("8. Execute and evaluate VAE")
    print("9. Train DeepSAD hyperparameters")
    print("10. Execute and evaluate DeepSAD")


def extract_matlab():
    """
        Extracts CSV files from PZTs

        Parameters: None
        Returns: None
    """

    # Convert .mat files to .csv
    folder_path = input("Enter the folder path of the Matlab files: ")
    Data_Preprocess.matToCsv(folder_path)
    print("Done")


# ----- Main -----
csv_dir = input("Enter the folder path of the CSV files: ")

# Main program loop
while True:

    # Display menu options and input selection
    main_menu()
    choice = input("Enter your choice: ")

    # Perform corresponding action
    # Quit program
    if choice == '0':
        print("Exiting...")
        quit()

    # Convert PZT files to CSV
    elif choice == '1':
        extract_matlab()

    # Execute signal processing transformations on data
    elif choice == '2':
        SP.saveFFT(csv_dir)
        SP.saveEMD(csv_dir)
        SP.saveHilbert(csv_dir)
        SP.saveSTFT(csv_dir)

    # Extract all statistical features and average across different paths
    elif choice == '3':
        saveFeatures(csv_dir)
        AverageFeatures(csv_dir)

    # Extract and save reduced features only
    elif choice == '4':
        FFT_HLB_Reduced_Feat(csv_dir)
        SP.saveFFTHLB(csv_dir)

    # Apply prognostic criteria to features
    elif choice == '5':
        features = retrieve_features(csv_dir)
        save_evaluation(features, "Features", csv_dir)

    #todo: should we get rid of PCA here
    elif choice == '6':
        savePCA(csv_dir)

    # Optimise hyperparameters for VAE
    elif choice == '7':
        conc_choice = input("Concatenate files? Y/N: ")
        vae_seed = int(input("Enter seed: "))
        if conc_choice.upper == "Y":
            hyperVAE(csv_dir, concatenate=True)
        elif conc_choice.upper == "N":
            hyperVAE(csv_dir)

    # Execute VAE and apply prognostic criteria
    elif choice == '8':
        #todo: just delete all this. Always plot graph and save HIs, decide whether validation or not
        save_graph_choice = input("Plot? Y/N: ")
        save_HI_choice = input("Save HI file? (needed for ensemble model) Y/N: ")
        valid_choice = input("Validation? Y/N: ")
        vae_seed = int(input("Enter seed: "))
        if save_graph_choice == "Y" or save_graph_choice == "y" and save_HI_choice == "Y" or save_HI_choice == "y":
            if valid_choice == "Y" or valid_choice == "y":
                saveVAE(csv_dir, save_graph=True, save_HI=True, valid=True)
            else:
                saveVAE(csv_dir, save_graph=True, save_HI=True, valid=False)
        if save_graph_choice == "Y" or save_graph_choice == "y" and save_HI_choice == "N" or save_HI_choice == "n":
            if valid_choice == "Y" or valid_choice == "y":
                saveVAE(csv_dir, save_graph=True, save_HI=False, valid=True)
            else:
                saveVAE(csv_dir, save_graph=True, save_HI=False, valid=False)
        if save_graph_choice == "N" or save_graph_choice == "n" and save_HI_choice == "Y" or save_HI_choice == "y":
            if valid_choice == "Y" or valid_choice == "y":
                saveVAE(csv_dir, save_graph=False, save_HI=True, valid=True)
            else:
                saveVAE(csv_dir, save_graph=False, save_HI=True, valid=False)
        if save_graph_choice == "N" or save_graph_choice == "n" and save_HI_choice == "N" or save_HI_choice == "n":
            if valid_choice == "Y" or valid_choice == "y":
                saveVAE(csv_dir, save_graph=False, save_HI=False, valid=True)
            else:
                saveVAE(csv_dir, save_graph=False, save_HI=False, valid=False)

    # Optimise hyperparameters for DeepSAD
    elif choice == '9':
        hyperDeepSad(csv_dir)

    # Execute DeepSAD and apply prognostic criteria
    elif choice == '10':
        saveDeepSAD(csv_dir)
        print("DeepSAD is completed for FFT and HLB")

    # In case of invalid input
    else:
        print("Invalid choice. Please select a valid option.")

