import numpy as np
import pandas as pd
import os

import Graphs
from Prognostic_criteria import fitness, test_fitness
from Graphs import HI_graph, big_plot

def wae(HIs, fold, filepath = "", name = ""):
    """
        Calculate Weighted Average Ensemble HIs

        Parameters:
        - HIs (2D np array): Matrix of HIs
        - filepath (string): Directory to save HI graph
        - fold (int): Test sample number
        - name (string): Name for HI graph (blank if no graph)

        Returns:
        - newHIs (1D np array): Fused HIs
    """

    freqs = HIs.shape[0]

    # Weights are equal to fitness scores
    weights = np.zeros((freqs))
    for run in range(freqs):
        weights[run] = fitness(np.concatenate((HIs[run][:][:fold], HIs[run][:][fold+1:])))[0]
    weights = weights/np.sum(weights)

    # New HIs for each of 5 samples
    newHIs = np.zeros((5, 30))
    for run in range(freqs):
        newHIs += weights[run] * HIs[run]

    # Save plot and return weighted HIs
    if name != "":
        HI_graph(newHIs, filepath, name, True)
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

    print("Loading " + type)
    if type == "VAE":
        HIs = []
        for repetition in range(len(seeds)):
            filename = f"{type}_{transform}_seed_{seeds[repetition]}.npy"
            HI = np.load(os.path.join(filepath, filename), allow_pickle=True)
            HI = HI.transpose(1, 0, 2, 3)
            HIs.append(HI)
        HIs = np.stack(HIs)

    else:
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

    meanFit = np.empty((freqnum, foldnum))
    stdFit = np.empty((freqnum, foldnum))
    testMeanFit = np.empty((freqnum, foldnum))
    testStdFit = np.empty((freqnum, foldnum))

    sumHIs = np.empty((freqnum, foldnum), dtype=object)
    sumTestHIs = np.empty((freqnum, foldnum), dtype=object)

    # Calculate fitness scores of HIs simple averaged across different seeds
    print("- Averaging between folds")
    for freq in range(freqnum):
        for fold in range(foldnum):
            HI_graph(meanHIs[freq][fold], filepath, f"{freqs[freq]}kHz_{type}_{transform}_{fold}", False)
            meanFit[freq, fold] = fitness(meanHIs[freq][fold])[0]
            sumHIs[freq, fold] = meanHIs[freq][fold] + stdHIs[freq][fold]
            stdFit[freq, fold] = abs(meanFit[freq][fold] - fitness(sumHIs[freq][fold])[0])

            testMeanFit[freq, fold] = test_fitness(meanHIs[freq][fold][fold], meanHIs[freq][fold])[0]
            sumTestHIs[freq, fold] = meanHIs[freq][fold][fold] + stdHIs[freq][fold][fold]
            testStdFit[freq, fold] = abs(testMeanFit[freq, fold] - test_fitness(sumTestHIs[freq, fold], sumHIs[freq, fold])[0])

    #Save to CSVs
    pd.DataFrame(meanFit).to_csv(os.path.join(filepath, f"meanfit_{type}_{transform}.csv"), index=False)
    pd.DataFrame(stdFit).to_csv(os.path.join(filepath, f"stdfit_{type}_{transform}.csv"), index=False)

    pd.DataFrame(testMeanFit).to_csv(os.path.join(filepath, f"test_meanfit_{type}_{transform}.csv"), index=False)
    pd.DataFrame(testStdFit).to_csv(os.path.join(filepath, f"test_stdfit_{type}_{transform}.csv"), index=False)


    # Carry out and save WAE F-all fitness between frequencies
    print("- WAE between frequencies")
    waeFit = []
    waeStdFit = []
    waeTestFit = []
    waeTestStdFit = []
    for fold in range(foldnum):
        waeHI = wae(meanHIs[:][fold], fold, filepath, f"WAE_{type}_{transform}_{fold}")
        waeFit.append(fitness(waeHI)[0])

        waeSumHI = wae(sumHIs[:][fold], fold)
        waeStdFit.append(abs(fitness(waeSumHI)[0]-waeFit[fold]))

        waeTestFit.append(test_fitness(waeHI[fold], waeHI)[0])
        waeTestStdFit.append(abs(test_fitness(waeSumHI[fold], waeSumHI)[0]-waeTestFit[fold]))
        
    pd.DataFrame(np.stack((waeFit, waeStdFit), axis=1)).to_csv(os.path.join(filepath, f"weighted_{type}_{transform}.csv"), index=False)
    pd.DataFrame(np.stack((waeTestFit, waeTestStdFit), axis=1)).to_csv(os.path.join(filepath, f"test_weighted_{type}_{transform}.csv"), index=False)

    print("- Plotting")
    big_plot(filepath, type, transform)

#csv_dir = r"C:\Users\pablo\Downloads\VAE_Ultimate_New"
csv_dir = r"C:\Users\Jamie\Documents\Uni\Year 2\Q3+4\Project\CSV-FFT-HLB-Reduced"
eval_wae(csv_dir, "DeepSAD", "FFT")
eval_wae(csv_dir, "DeepSAD", "HLB")