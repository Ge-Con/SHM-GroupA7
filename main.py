# Import external libraries
import pandas as pd
import numpy as np
import os

# Import modules
import extract_features
import PCA
from Signal_Processing import Data_Preprocess
from prognosticcriteria_v2 import fitness
from DeepSAD import DeepSAD_train_run, plot_ds_images
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
np.set_printoptions(linewidth=400)
np.set_printoptions(precision=4, suppress=True)


def saveFeatures(dir):
    """
        Feature Extraction. This function processes files with names ending in 'kHz.csv', 'FFT.csv', 'HLB.csv',
        'EMD.csv' and 'SFT.csv' and saves the features in a new .csv file with '_FT' appended before the extension.

        Parameters:
            dir (str): The directory path containing CSV files
        Returns: None
    """

    # Extract and save features for each frequency
    print("Extracting Features:...")
    for root, dirs, files in os.walk(dir):
        for name in files:

            # Time domain
            if name.endswith('kHz.csv'):
                # Read data - don't put next line outside if statement otherwise invalid files will be read
                data = pd.read_csv(os.path.join(root, name))

                # Extract and save features
                features = extract_features.time_to_feature(data)
                new_filename = name.replace('kHz.csv', 'kHz_FT.csv')
                csv_file_path = os.path.join(root, new_filename)
                features.to_csv(csv_file_path, index=False)

            # FFT
            elif name.endswith('FFT.csv'):
                data = pd.read_csv(os.path.join(root, name))
                features = extract_features.freq_to_feature(data)
                new_filename = name.replace('FFT.csv', 'FFT_FT.csv')
                csv_file_path = os.path.join(root, new_filename)
                features.to_csv(csv_file_path, index=False)

            # Hilbert transform
            elif name.endswith('HLB.csv'):
                data = pd.read_csv(os.path.join(root, name))
                features = extract_features.time_to_feature(data)
                new_filename = name.replace('HLB.csv', 'HLB_FT.csv')
                csv_file_path = os.path.join(root, new_filename)
                features.to_csv(csv_file_path, index=False)

            # Empirical Mode Decomposition (EMD)
            elif name.endswith('EMD.csv'):
                data = pd.read_csv(os.path.join(root, name))
                features = extract_features.time_to_feature(data)
                new_filename = name.replace('EMD.csv', 'EMD_FT.csv')
                csv_file_path = os.path.join(root, new_filename)
                features.to_csv(csv_file_path, index=False)

            # STFT
            elif name.endswith('SFT.csv'):
                # Read data
                data = pd.read_csv(os.path.join(root, name))
                # Unflatten to 3D array
                data3d = [[[0 for _ in range(17)] for _ in range(126)] for _ in range(56)]
                for k in range(56):
                    for i in range(126):
                        for j in range(17):
                            data3d[k][i][j] = data.iloc[126*k + i, j]

                # Extract and save features
                features = extract_features.STFT_to_feature(data3d)
                new_filename = name.replace('SFT.csv', 'SFT_FT.csv')
                csv_file_path = os.path.join(root, new_filename)
                features.to_csv(csv_file_path, index=False)


def FFT_HLB_Reduced_Feat(dir):
    """
        Save reduced features for FFT and Hilbert transforms

        Parameters:
            dir(str): The directory path containing CSV files
        Returns: None
    """

    print("Extracting Features:...")
    for root, dirs, files in os.walk(dir):
        for name in files:

            # FFT
            if name.endswith('FFT.csv'):
                # Read transform and extract reduced features
                data = pd.read_csv(os.path.join(root, name))
                features = extract_features.FFT_Feat_reduced(data)

                # Save to CSV file
                new_filename = name.replace('FFT.csv', 'FFT_FT_Reduced.csv')
                csv_file_path = os.path.join(root, new_filename)
                features.to_csv(csv_file_path, index=False)

            # Hilbert transform
            elif name.endswith('HLB.csv'):
                # Read transform and extract reduced features
                data = pd.read_csv(os.path.join(root, name))
                features = extract_features.HLB_Feat_reduced(data)

                # Save to CSV file
                new_filename = name.replace('HLB.csv', 'HLB_FT_Reduced.csv')
                csv_file_path = os.path.join(root, new_filename)
                features.to_csv(csv_file_path, index=False)


def AverageFeatures(rootdir):
    """
        Averages and saves extracted features across paths

        Parameters:
            rootdir(str): The directory path containing CSV files
        Returns: None
    """

    frequencies = ["050", "100", "125", "150", "200", "250"]
    samples = ["PZT-CSV-L1-03", "PZT-CSV-L1-04", "PZT-CSV-L1-05", "PZT-CSV-L1-09", "PZT-CSV-L1-23"]

    print("Combining Features:...")
    # Walk folders of each sample in turn
    for sample in samples:
        dir = os.path.join(rootdir, sample)

        # Create All Features (AF) files
        for root, dirs, files in os.walk(dir):
            all_feat = np.empty((6, 5), dtype=object)
            flag = False  # Flag if any useful files were found to prevent errors if arrays remain empty

            for name in files:
                for freq in frequencies:
                    # Read and add to correct position in all_feat array
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

            #  Save features found to separate folder as AF
            if flag:  # If at least one file was the correct type
                for freq in ["050", "100", "125", "150", "200", "250"]:
                    # Concatenate array to one less dimension
                    combinedfeatures = np.concatenate(
                        [all_feat[frequencies.index(freq), i] for i in range(all_feat[0].shape[0])], axis=0)
                    # Save to new directory
                    root_new = root.replace("PZT", "PZT-ONLY-FEATURES")
                    if not os.path.exists(root_new):
                        os.makedirs(root_new)

                    # Save CSV files
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
                    data_mean = np.mean(data, axis=0)  # Corrected to mean over the correct axis
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


def correlateFeatures(rootdir):
    """
        Correlates extracted average features, saving correlation matrix and reduced feature matrices.
        This is no longer used in the framework, but is kept for completeness.

        Parameters:
            rootdir(str): The directory path containing CSV files
        Returns: None
    """

    frequencies = ["050", "100", "125", "150", "200", "250"]
    samples = ["PZT-CSV-L1-03", "PZT-CSV-L1-04", "PZT-CSV-L1-05", "PZT-CSV-L1-09", "PZT-CSV-L1-23"]

    # Walk folders of each sample in turn
    for sample in samples:
        dir = rootdir + "\\" + sample

        # Read All Features for each state and calculate Mean Features
        print("Reading Files: " + sample)
        meanfeatures = np.empty((6), dtype=object)
        for root, dirs, files in os.walk(dir):
            for name in files:
                if name.endswith("kHz_AF.csv"):
                    data = np.array(pd.read_csv(os.path.join(root, name)))
                    data = np.mean(data, axis=1)
                    if str(type(meanfeatures[frequencies.index(name[:3])])) == "<class 'NoneType'>":    # First to be added
                        meanfeatures[frequencies.index(name[:3])] = np.array([data])
                    else:   # Concatenate subsequent entries
                        meanfeatures[frequencies.index(name[:3])] = np.concatenate((meanfeatures[frequencies.index(name[:3])], np.array([data])))

        # Correlate mean features
        print("Correlating Features:...")
        alldelete = np.empty((6), dtype=object)
        for freq in frequencies:
            csv_file_path = os.path.join(dir, freq + "kHz_MF.csv") # MF = Mean Features
            # Read mean features from file
            pd.DataFrame(meanfeatures[frequencies.index(freq)]).to_csv(csv_file_path, index=False)

            #Use feature_correlation to return the correlation matrix, reduced matrix and features deleted
            correlation_matrix, features, to_delete = extract_features.feature_correlation(meanfeatures[frequencies.index(freq)])

            #Save all to files
            alldelete[frequencies.index(freq)] = to_delete
            csv_file_path = os.path.join(dir, freq + "kHz_CF.csv") # CM = Correlated feature matrix
            pd.DataFrame(correlation_matrix).to_csv(csv_file_path, index=False)
            csv_file_path = os.path.join(dir, freq + "kHz_RF.csv") # RF = Remainder features
            pd.DataFrame(features).to_csv(csv_file_path, index=False)

        # Save lists of deleted features for inspection
        csv_file_path = os.path.join(dir, "DF.csv") # DF = Deleted features
        pd.DataFrame(alldelete).to_csv(csv_file_path, index=False)


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
        Save basic CSVs and graphs of prognostic criteria applied to HIs

        Parameters:
            features (3D np array): feature extraction or HI output of all samples - 2D or 3D
            label (str): Name of framework to save results under
            dir (str): Directory to save results to
            files_used (list): Names of features used
        Returns:
            criteria (3D+ array): Prognostic criteria results with 4 PC types, 5 frequencies,
                                  n features - extra dimensions if recursive calls
    """

    # Create save directory for HIs which will not overwrite existing files
    count = 1
    savedir = dir + '\\' + label
    while os.path.exists(savedir + '.npy'):
        savedir = dir + '\\' + label + str(count)
        count += 1

    # Save HIs to .npy file for use by weighted average ensemble (WAE)
    np.save(savedir, features)

    # Recursively call this function if a list of sets of HIs has been passed
    if features.ndim > 4:
        criteria = []
        for i in range(features.shape[1]):
            criteria.append(save_evaluation(features[:, i], label + " framework " + str(i+1), dir, files_used))
    else:

        frequencies = ["050", "100", "125", "150", "200", "250"]
        # Initiliase arrays for feature extraction results, for fitness and the three criteria respectively
        criteria = np.empty((4, 6, len(features[0])))
        # Iterate through each frequency and calculate features
        featuresonly = files_used[0] == ""

        # Evaluate prognostic criteria for each frequency
        for freq in range(6):
            print("Evaluating: " + frequencies[freq] + "kHz")
            for feat in range(len(features[0])):
                if feat % 50 == 0 and feat != 0:    # Print occasional progress updates
                    print(feat, " features")

                # Evaluate and store prognostic criteria
                features[freq][feat] = np.array(features[freq][feat])
                ftn, mo, tr, pr, error = fitness(features[freq][feat])
                criteria[0][freq][feat] = float(ftn)
                criteria[1][freq][feat] = float(mo)
                criteria[2][freq][feat] = float(pr)
                criteria[3][freq][feat] = float(tr)

        # Save all to CSV files
        print("Saving to CSVs")
        pd.DataFrame(criteria[0]).to_csv(os.path.join(dir, label + " Fit.csv"), index=False)  # Feature against frequency
        pd.DataFrame(criteria[1]).to_csv(os.path.join(dir, label + " Mon.csv"), index=False)
        pd.DataFrame(criteria[2]).to_csv(os.path.join(dir, label + " Tre.csv"), index=False)
        pd.DataFrame(criteria[3]).to_csv(os.path.join(dir, label + " Pro.csv"), index=False)

        # Graph HIs
        for freq in range(6):
            print("Graphing: " + frequencies[freq] + "kHz")
            for feat in range(len(features[0])):
                Graphs.HI_graph(features[freq][feat], dir=dir, name=f"{label}-{frequencies[freq]}-{feat}")

            # Make bar chart of prognostic criteria against feature
            if featuresonly:
                files_used = np.array([str(i) for i in range(len(features[0]))])
            try:
                Graphs.criteria_chart(files_used, criteria[1][freq], criteria[2][freq], criteria[3][freq], dir=dir, name=f"{label}-{frequencies[freq]}")
            except ValueError:  # In case the wrong number of labels was given
                print("ValueError, bar chart not generated")

        # Bar charts against frequency
        #for feat in range(len(features[0])):
        #    Graphs.criteria_chart(frequencies, criteria[1][:, feat], criteria[2][:, feat], criteria[3][:, feat], dir=dir, name=label + "-" + str(feat))

        # Simple average of prognostic criteria for features across frequencies
        if featuresonly:
            avs = np.empty((4, 2), dtype=object)
            for crit in range(4):
                avs[crit, 0] = np.expand_dims(np.mean(criteria[crit], axis= 0),axis=0)[0]
                avs[crit, 1] = np.std(criteria[crit], axis = 0)
            Graphs.criteria_chart(files_used, avs[1][0], avs[2][0], avs[3][0], dir=dir, name=f"{label}-Av")
            av_arr = np.vstack((avs[0, 0], avs[0, 1]))
            # Save to CSV file
            pd.DataFrame(av_arr).to_csv(os.path.join(dir, label + " Fit AF.csv"), index=False)

        return criteria

def retrieve_features(dir):
    """
        Read features from CSV files to numpy array

        Parameters:
        - dir (str): CSV root folder directory
        Returns:
        - features (4D np array): Signal features, 6 frequencies x 139 features x 5 samples x n states
    """

    frequencies = ["050", "100", "125", "150", "200", "250"]
    features = np.empty((6, 139), dtype=object)

    # Files will be read in order of directory
    for root, dirs, files in os.walk(dir):
        for name in files:
            # Continue only with Mean Features files
            if name.endswith("MF.csv"):
                # Read file data, and determine frequency from filename
                data = np.array(pd.read_csv(os.path.join(root, name))).transpose()
                freq = frequencies.index(name[:3])

                # Add each feature to correct location in features array
                for feat in range(139):
                    if str(type(features[freq][feat])) == "<class 'NoneType'>": # First item in location
                        features[freq][feat] = np.array([scale_exact(data[feat])])
                    else:   # Stack subsequent items
                        features[freq][feat] = np.vstack([features[freq][feat], scale_exact(data[feat])])

    return features


def hyperDeepSad(dir):
    """
        Optimise hyperparameters for DeepSAD

        Parameters:
        - dir (str): CSV root folder directory
        Returns: None
    """
    filenames = ["FFT_FT_Reduced", "HLB_FT_Reduced"]
    samples = ["PZT-FFT-HLB-L1-03", "PZT-FFT-HLB-L1-04", "PZT-FFT-HLB-L1-05", "PZT-FFT-HLB-L1-09", "PZT-FFT-HLB-L1-23"]
    frequencies = ["050", "100", "125", "150", "200", "250"]
    for file in filenames:
        for freq in frequencies:
            # Optimise hyperparameters
            params = DeepSAD_train_run(dir, freq, file, True)
            # Save to external file
            for sample in range(5):
                HYP.simple_store_hyperparameters(params[sample], file, samples[sample], freq, dir)


def saveDeepSAD(dir):
    """
        Run and save DeepSAD HIs

        Parameters:
        - dir (str): CSV root folder directory
        Returns: None
    """

    frequencies = ["050", "100", "125", "150", "200", "250"]
    filename_FFT = "FFT_FT_Reduced"
    filename_HLB = "HLB_FT_Reduced"    # No need for .csv

    # Initialise HI arrays
    HIs_FFT = np.empty((6), dtype=object)
    HIs_HLB = np.empty((6), dtype=object)

    # FFT features for each frequency
    for freq in range(len(frequencies)):
        print(f"Processing frequency: {frequencies[freq]} kHz for FFT")
        HIs_FFT[freq] = DeepSAD_train_run(dir, frequencies[freq], filename_FFT)
    # Save and plot results
    save_evaluation(np.array(HIs_FFT), "DeepSAD_FFT", dir, filename_FFT)
    plot_ds_images(dir, "FFT")

    # Hilbert features for each frequency
    for freq in range(len(frequencies)):
        print(f"Processing frequency: {frequencies[freq]} kHz for HLB")
        HIs_HLB[freq] = DeepSAD_train_run(dir, frequencies[freq], filename_HLB)
    # Save and plot results
    save_evaluation(np.array(HIs_HLB), "DeepSAD_HLB", dir, filename_HLB)
    plot_ds_images(dir, "HLB")


def hyperVAE(dir, concatenate=False):
    """
        Optimise hyperparameters for VAE

        Parameters:
        - dir (str): CSV root folder directory
        Returns: None

        This function is a work in progress, VAE is currently run independently
    """
    global vae_seed

    tf.compat.v1.reset_default_graph()
    tf.random.set_seed(vae_seed)
    np.random.seed(vae_seed)

    filenames = ["FFT_FT_Reduced", "HLB_FT_Reduced"]
    samples = ["PZT-FFT-HLB-L1-03", "PZT-FFT-HLB-L1-04", "PZT-FFT-HLB-L1-05", "PZT-FFT-HLB-L1-09", "PZT-FFT-HLB-L1-23"]
    panels = ("L103", "L105", "L109", "L104", "L123")
    freqs = ("050_kHz", "100_kHz", "125_kHz", "150_kHz", "200_kHz", "250_kHz")

    # Concatenate input files ready for VAE
    if concatenate:
        for panel in samples:
            for file in filenames:
                process_csv_files(dir, panel, file)
    # Optimise hyperparameters
    for file_type in filenames:
        counter = 0
        for panel in panels:
            for freq in freqs:

                train_filenames = []

                for i in tuple(x for x in panels if x != panel):
                    filename = os.path.join(dir, f"concatenated_{freq}_{i}_{file_type}.csv")
                    train_filenames.append(filename)

                counter += 1
                print("Counter: ", counter)
                print("Panel: ", panel)
                print("Freq: ", freq)
                print("SP Features: ", file_type)

                # For vae_train_data, create the merged file with all 4 panels, delete the last column. I don't know why we delete the last column though
                vae_train_data, flags = HYPparameters.mergedata(train_filenames)
                vae_train_data.drop(vae_train_data.columns[len(vae_train_data.columns) - 1], axis=1, inplace=True)

                # Same as with train data. Read the filename, and delete the last column
                # Also hardcoded
                test_filename = os.path.join(dir, f"concatenated_{freq}_{panel}_{file_type}.csv")
                vae_test_data = pd.read_csv(test_filename, header=None).values.transpose()
                vae_test_data = np.delete(vae_test_data, -1, axis=1)

                # Normalizing the train and test data, scaled with vae_train_data
                vae_scaler = HYPparameters.StandardScaler()
                vae_scaler.fit(vae_train_data)
                vae_train_data = vae_scaler.transform(vae_train_data)
                vae_test_data = vae_scaler.transform(vae_test_data)

                # Applying PCA to the train and test data, fit with vae_train_data
                vae_pca = HYPparameters.PCA(n_components=30)
                vae_pca.fit(vae_train_data)
                vae_train_data = vae_pca.transform(vae_train_data)
                vae_test_data = vae_pca.transform(vae_test_data)

                hyperparameters = HYPparameters.hyperparameter_optimisation(vae_train_data, vae_test_data, vae_scaler, vae_pca,
                                                              vae_seed, file_type, panel, freq, dir, n_calls=40)
                HYPparameters.simple_store_hyperparameters(hyperparameters, file_type, panel, freq, dir)

def saveVAE(dir, save_graph=True, save_HI=True, valid=False):
    """
        Run and save VAE HIs
    
        Parameters:
        - dir (str): CSV root folder directory
        Returns: None

        This function is a work in progress, VAE is currently run independently
    """
    global vae_seed

    tf.compat.v1.reset_default_graph()
    tf.random.set_seed(vae_seed)
    np.random.seed(vae_seed)

    panels = ("L103", "L105", "L109", "L104", "L123")
    freqs = ("050_kHz", "100_kHz", "125_kHz", "150_kHz", "200_kHz", "250_kHz")
    filenames = ["FFT_FT_Reduced", "HLB_FT_Reduced"]
    colors = ("b", "g", "y", "r", "m")

    time_steps = 30
    num_HIs = 5  # Number of rows in f_all_array
    num_freqs = len(freqs)
    num_panels = len(panels)

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

                fitness_test = HYPparameters.test_fitness(health_indicators[2], health_indicators[1])
                print("Fitness test", fitness_test)

                if valid:
                    fitness_valid = HYPparameters.test_fitness(health_indicators[6], health_indicators[1])
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
    folder_path = input("Enter the folder path of the Matlab files: ")
    Data_Preprocess.matToCsv(folder_path)
    print("Done")


# ----- Main -----
csv_dir = input("Enter the folder path of the CSV files: ")

# Main program loop
while True:
    main_menu()
    choice = input("Enter your choice: ")

    if choice == '0':
        print("Exiting...")
        quit()
    elif choice == '1':
        extract_matlab()
    elif choice == '2':
        SP.saveFFT(csv_dir)
        SP.saveEMD(csv_dir)
        SP.saveHilbert(csv_dir)
        SP.saveSTFT(csv_dir)
    elif choice == '3':
        saveFeatures(csv_dir)
        AverageFeatures(csv_dir)
    elif choice == '4':
        FFT_HLB_Reduced_Feat(csv_dir)
        SP.saveFFTHLB(csv_dir)
    elif choice == '5':
        features = retrieve_features(csv_dir)
        save_evaluation(features, "Features", csv_dir)
    elif choice == '6':
        savePCA(csv_dir)
    elif choice == '7':
        conc_choice = input("Concatenate files? Y/N: ")
        vae_seed = int(input("Enter seed: "))
        if conc_choice == "Y" or conc_choice == "y":
            hyperVAE(csv_dir, concatenate=True)

        if conc_choice == "N" or conc_choice == "n":
            hyperVAE(csv_dir)
    elif choice == '8':
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
    elif choice == '9':
        hyperDeepSad(csv_dir)
    elif choice == '10':
        saveDeepSAD(csv_dir)
    else:
        print("Invalid choice. Please select a valid option.")