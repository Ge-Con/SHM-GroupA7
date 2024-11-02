import numpy as np
from Prognostic_criteria import fitness, Mo_single, Tr, Pr_single
import pandas as pd
import os

def test_fitness(HIs, test_HI):
    """
        Calculate test fitness score
        
        Parameters:
        - HIs (2D np array): Matrix of HIs
        - test_HI (1D np array): List of HI for test panel
        
        Returns:
        - fitness (float): F-test score
    """
    
    monotonicity = Mo_single(HIs)
    trendability = Tr(np.vstack([HIs, test_HI]))
    prognosability = Pr_single(HIs, test_HI)
    
    return monotonicity + trendability + prognosability

def wae(HIs):
    """
        Calculate Weighted Average Ensemble HIs

        Parameters:
        - HIs (2D np array): Matrix of HIs

        Returns:
        - newHIs (1D np array): Fused HIs
    """

    num = HIs.shape[0]
    #Array of 6 HIs
    w = np.zeros((num))
    for run in range(num):
        w[run] = fitness(HIs[run])[0]
    w = w/np.sum(w)
    newHIs = np.zeros((5, 30))
    for run in range(num):
        newHIs += w[run] * HIs[run]
    return newHIs


print("Files must be 5D - 6 frequencies, n repetitions, 5 folds, 5 panels of 30 HIs")

def eval_wae(filepath):
    """
        Execute and evaluate WAE HIs
        This function was written shortly before the TAS deadline so may contain errors, in addition to the following:
        - The function must be manually changed between producing results for FFT and Hilbert transform
        - Standard deviation for AI methods could not be implemented due to numpy errors.
        The hurried writing was also the source for often poor choice in variable names.

        Parameters:
        - merge (bool): Whether to perform WAE across folds

        Returns: None
    """

    # Seeds used
    files = ["140"]#["42", "110", "120", "130", "140"]    # 5 seeds for VAE, 3 for DeepSAD due to time

    # Simple averaging repetitions
    storeHIs = np.empty((3), dtype=object)
    for repetition in range(3):
        print("Repetition " + str(repetition+1))
        filename = "DF" + files[repetition] + ".npy"
        storeHIs[repetition] = np.load(filepath + "\\" + filename, allow_pickle=True)

    # Define arrays for mean and standard deviation of HIs and fitness scores
    meanHIs = np.mean(storeHIs, axis=0)
    #stdHIs = np.std(storeHIs, axis=0)
    meanfit = np.empty((6, 5))
    #stdfit = np.empty((6, 5))

    # Calculate F-all scores between repetitions
    for fold in range(5):
        for freq in range(6):
            meanfit[freq, fold] = fitness(meanHIs[freq][fold])[0]
            #stdfit[freq, fold] = meanfit[freq][fold] - fitness(meanHIs[freq][fold]+stdHIs[freq][fold])[0]

    #Save to CSVs
    pd.DataFrame(meanfit).to_csv(os.path.join(filepath, "meanfitFFT.csv"), index=False)
    #pd.DataFrame(meanfit).to_csv(os.path.join(dir, "stdfit.csv"), index=False)

    # Repeat for F-test scores
    print("F-test:")
    tmeanfit = np.empty((6, 5))
    #tstdfit = np.empty((5, 6))

    for fold in range(5):
        for freq in range(6):
            tmeanfit[freq, fold] = test_fitness(meanHIs[freq][fold][fold], meanHIs[freq][fold])
            #tstdfit[freq, fold] = tmeanfit[freq, fold] - test_fitness(meanHIs[freq, fold, fold] + stdHIs[freq, fold, fold], meanHIs[freq, fold] + stdHIs[freq, fold])[0]
    pd.DataFrame(tmeanfit).to_csv(os.path.join(filepath, "tmeanfitFFT.csv"), index=False)
    #pd.DataFrame(meanfit).to_csv(os.path.join(dir, "tstdfit.csv"), index=False)

    # Carry out and save WAE between folds for F-all
    newHIses = []
    for fold in range(5):
        print("-> Fold " + str(fold))
        newHIs = wae(meanHIs[:][fold])  # Average repetitions
        newHIses.append(fitness(newHIs)[0])
    pd.DataFrame(newHIses).to_csv(os.path.join(filepath, "weightedFFT.csv"), index=False)

    # Carry out and save WAE between folds for F-test
    newHIses = []
    for fold in range(5):
        print("-> Fold " + str(fold))
        newHIs = wae(meanHIs[:][fold])  # Average repetitions
        newHIses.append(test_fitness(newHIs[fold], newHIs))
    pd.DataFrame(newHIses).to_csv(os.path.join(filepath, "tweightedFFT.csv"), index=False)