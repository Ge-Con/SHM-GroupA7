import numpy as np
import pandas as pd
import os

from Prognostic_criteria import fitness, test_fitness
from Graphs import HI_graph

def wae(HIs, filepath, name):
    """
        Calculate Weighted Average Ensemble HIs

        Parameters:
        - HIs (2D np array): Matrix of HIs
        - filepath (string): Directory to save HI graph
        - name (string): Name for HI graph

        Returns:
        - newHIs (1D np array): Fused HIs
    """

    freqs = HIs.shape[0]

    # Weights are equal to fitness scores
    weights = np.zeros((freqs))
    for run in range(freqs):
        weights[run] = fitness(HIs[run])[0]
    weights = weights/np.sum(weights)

    # New HIs for each of 5 samples
    newHIs = np.zeros((5, 30))
    for run in range(freqs):
        newHIs += weights[run] * HIs[run]

    # Save plot and return weighted HIs
    HI_graph(newHIs, filepath, "WAE_" + name)
    return newHIs


print("Files must be 5D - 6 frequencies, n repetitions, 5 folds, 5 panels of 30 HIs")

def eval_wae(filepath, type):
    """
        Execute and evaluate mean HIs and weighted average ensemble learning for HIs
        Standard deviation not yet implemented due to numpy errors

        Parameters:
        - filepath (string): directory of HI CSVs
        - type (string): "FFT" or "HLB"

        Returns: None
    """

    # Seeds used
    seeds = ["42", "110", "120", "130", "140"]
    print("Ensure .npy filenames end with '_n', where n is from 1 to the number of seeds used")
    print("\te.g. DeepSAD_FFT_1.npy")
    print("\tThis should match the number of seeds given in WAE.py")
    print("\tNumber of seeds given: " + str(len(seeds)))

    # Repetitions are the same HIs generated with different seeds
    # Simple averaging repetitions
    HIs = np.empty((len(seeds)), dtype=object)
    for repetition in range(len(seeds)):
        print("Repetition " + str(repetition+1))
        filename = "DeepSAD_" + type + "_" + seeds[repetition] + ".npy"
        HIs[repetition] = np.load(filepath + "\\" + filename, allow_pickle=True)

    # Define arrays for mean and standard deviation of HIs and fitness scores between seeds
    meanHIs = np.mean(HIs, axis=0)
    stdHIs = np.std(HIs, axis=0)

    meanfit = np.empty((6, 5))
    stdfit = np.empty((6, 5))

    # Calculate fitness scores of HIs simple averaged across different seeds
    # Calculate F-all scores between repetitions
    for fold in range(5):
        for freq in range(6):
            meanfit[freq, fold] = fitness(meanHIs[freq][fold])[0]
            #stdfit[freq, fold] = meanfit[freq][fold] - fitness(meanHIs[freq][fold]+stdHIs[freq][fold])[0]
            #TODO: Fix std deviation once we have data

    #Save to CSVs
    pd.DataFrame(meanfit).to_csv(os.path.join(filepath, "meanfit_" + type + ".csv"), index=False)
    pd.DataFrame(stdfit).to_csv(os.path.join(filepath, "stdfit_" + type + ".csv"), index=False)

    # Repeat for F-test scores
    print("F-test:")
    testMeanfit = np.empty((6, 5))
    testStdfit = np.empty((6, 5))

    for fold in range(5):
        for freq in range(6):
            testMeanfit[freq, fold] = test_fitness(meanHIs[freq][fold], meanHIs[freq][fold][fold])
            #testStdfit[freq, fold] = testMeanfit[freq, fold] - test_fitness(meanHIs[freq, fold, fold] + stdHIs[freq, fold, fold], meanHIs[freq, fold] + stdHIs[freq, fold])[0]
    pd.DataFrame(testMeanfit).to_csv(os.path.join(filepath, "test_meanfit_" + type + ".csv"), index=False)
    pd.DataFrame(testStdfit).to_csv(os.path.join(filepath, "test_stdfit_" + type + ".csv"), index=False)


    # Carry out and save WAE F-all fitness between frequencies
    waeHIs = []
    for fold in range(5):
        print("-> Fold " + str(fold))
        waeHI = wae(meanHIs[:][fold], filepath, type)  # Average repetitions
        waeHIs.append(fitness(waeHI)[0])
    pd.DataFrame(waeHIs).to_csv(os.path.join(filepath, "weighted_" + type + ".csv"), index=False)

    # Carry out and save WAE F-test fitness between frequencies
    waeHIs = []
    for fold in range(5):
        print("-> Fold " + str(fold))
        waeHI = wae(meanHIs[:][fold], filepath, type)  # Average repetitions
        waeHIs.append(test_fitness(waeHI[fold], waeHI))
    pd.DataFrame(waeHIs).to_csv(os.path.join(filepath, "test_weighted_" + type + ".csv"), index=False)