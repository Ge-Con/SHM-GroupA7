import numpy as np
import pandas as pd
import os

import Graphs
from Prognostic_criteria import fitness, test_fitness
from Graphs import HI_graph, big_plot

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
    HI_graph(newHIs, filepath, name, False)
    return newHIs


#print("Files must be 5D - 6 frequencies, n repetitions, 5 folds, 5 panels of 30 HIs")

def eval_wae(filepath, type, transform):
    """
        Execute and evaluate mean HIs and weighted average ensemble learning for HIs
        Standard deviation not yet implemented due to numpy errors

        Parameters:
        - filepath (string): directory of HI CSVs
        - type (string): "DeepSAD" or "VAE"
        - transform (string): "FFT" or "HLB"

        Returns: None
    """

    # Seeds used
    seeds = ("42", "52", "62", "72", "82")
    freqs = ("050", "100", "125", "150", "200", "250")

    # Repetitions are the same HIs generated with different seeds
    # Simple averaging repetitions
    HIs = []
    for repetition in range(len(seeds)):
        filename = f"{type}_{transform}_seed_{seeds[repetition]}.npy"
        HIs.append(np.load(os.path.join(filepath, filename), allow_pickle=True))
    HIs = np.stack(HIs)

    meanHIs = []
    stdHIs = []
    for freq in range(HIs.shape[1]):
        meanHIs.append(np.mean(HIs[:, freq], axis=0))
        stdHIs.append(np.std(HIs[:, freq], axis=0))

    freqnum = HIs.shape[1]
    foldnum = HIs[0][0].shape[0]

    meanfit = np.empty((freqnum, foldnum))
    stdfit = np.empty((freqnum, foldnum))

    # Calculate fitness scores of HIs simple averaged across different seeds
    # Calculate F-all scores between repetitions
    for freq in range(freqnum):
        for fold in range(foldnum):
            HI_graph(meanHIs[freq][fold], filepath, f"{freqs[freq]}kHz_{type}_{transform}_{fold}", False)
            meanfit[freq, fold] = fitness(meanHIs[freq][fold])[0]
            stdfit[freq, fold] = meanfit[freq][fold] - fitness(meanHIs[freq][fold]+stdHIs[freq][fold])[0]

    #Save to CSVs
    pd.DataFrame(meanfit).to_csv(os.path.join(filepath, f"meanfit_{type}_{transform}.csv"), index=False)
    pd.DataFrame(stdfit).to_csv(os.path.join(filepath, f"stdfit_{type}_{transform}.csv"), index=False)

    # Repeat for F-test scores
    testMeanfit = np.empty((freqnum, foldnum))
    testStdfit = np.empty((freqnum, foldnum))

    for freq in range(freqnum):
        for fold in range(foldnum):
            testMeanfit[freq, fold] = test_fitness(meanHIs[freq][fold][fold], meanHIs[freq][fold])[0]
            testStdfit[freq, fold] = testMeanfit[freq, fold] - test_fitness(meanHIs[freq][fold][fold] + stdHIs[freq][fold][fold], meanHIs[freq][fold] + stdHIs[freq][fold])[0]
    pd.DataFrame(testMeanfit).to_csv(os.path.join(filepath, f"test_meanfit_{type}_{transform}.csv"), index=False)
    pd.DataFrame(testStdfit).to_csv(os.path.join(filepath, f"test_stdfit_{type}_{transform}.csv"), index=False)


    # Carry out and save WAE F-all fitness between frequencies
    waeFit = []
    for fold in range(foldnum):
        waeHI = wae(meanHIs[:][fold], filepath, f"WAE_{type}_{transform}_{fold}")
        waeFit.append(fitness(waeHI))
    pd.DataFrame(waeFit).to_csv(os.path.join(filepath, f"test_weighted_{type}_{transform}.csv"), index=False)

    # Carry out and save WAE F-test fitness between frequencies
    waeFit = []
    for fold in range(foldnum):
        waeHI = wae(meanHIs[:][fold], filepath, f"WAE_{type}_{transform}_{fold}")
        waeFit.append(test_fitness(waeHI[fold], waeHI))
    pd.DataFrame(waeFit).to_csv(os.path.join(filepath, f"test_weighted_{type}_{transform}.csv"), index=False)

    big_plot(filepath, type, transform)

csv_dir = "C:\\Users\\Jamie\\Documents\\Uni\\Year 2\\Q3+4\\Project\\CSV-FFT-HLB-Reduced"
eval_wae(csv_dir, "DeepSAD", "FFT")
eval_wae(csv_dir, "DeepSAD", "HLB")