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

    freqnum = HIs.shape[0]

    # Weights are equal to fitness scores
    weights = np.zeros((freqnum))
    for freq in range(freqnum):
        weights[freq] = fitness(np.concatenate((HIs[freq, :fold], HIs[freq, fold+1:])))[0]
    weights = weights/np.sum(weights)

    # New HIs for each of 5 samples
    newHIs = np.zeros((5, 30))
    for freq in range(freqnum):
        newHIs += weights[freq] * HIs[freq]

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
    # Repetitions are the same HIs generated with different seeds
    # Number of repetitions
    repnum = len(seeds)

    # Load HI data
    print(f"Loading {type} - {transform}")
    if type == "VAE":
        HIs = []
        for rep in range(repnum):
            filename = f"{type}_{transform}_seed_{seeds[rep]}.npy"
            HI = np.load(os.path.join(filepath, filename), allow_pickle=True)
            HIs.append(HI)
        HIs = np.stack(HIs)

    else:
        HIs = []
        for rep in range(repnum):
            filename = f"{type}_{transform}_seed_{seeds[rep]}.npy"
            HI = np.stack(np.load(os.path.join(filepath, filename), allow_pickle=True))
            HIs.append(HI)
        HIs = np.stack(HIs)
        HIs.transpose(0, 2, 1, 3, 4)    # Repetition, frequency, fold, specimen, HIs

    # Determine dimensions
    freqnum = HIs.shape[1]
    foldnum = HIs[0, 0].shape[0]

    # print(HIs.shape)
    # print(freqnum)
    # print(foldnum)

    # Calculate mean and standard deviation of fitness scores
    print("- Fitness")
    fall = np.empty((repnum, freqnum, foldnum))
    ftest = np.empty((repnum, freqnum, foldnum))
    mean_fall = np.empty((freqnum, foldnum))
    mean_ftest = np.empty((freqnum, foldnum))
    std_fall = np.empty((freqnum, foldnum))
    std_ftest = np.empty((freqnum, foldnum))
    for fold in range(foldnum):
        for freq in range(freqnum):
            for rep in range(repnum):
                fall[rep, freq, fold] = fitness(HIs[rep, freq, fold])[0]
                ftest[rep, freq, fold] = test_fitness(HIs[rep, freq, fold, fold], np.concatenate((HIs[rep, freq, fold, :fold], HIs[rep, freq, fold, fold+1:])))[0]
            mean_fall[freq, fold] = np.mean(fall[:, freq, fold])
            std_fall[freq, fold] = np.std(fall[:, freq, fold])
            mean_ftest[freq, fold] = np.mean(ftest[:, freq, fold])
            std_ftest[freq, fold] = np.std(ftest[:, freq, fold])

    pd.DataFrame(mean_fall).to_csv(os.path.join(filepath, f"meanfit_{type}_{transform}.csv"), index=False)
    pd.DataFrame(std_fall).to_csv(os.path.join(filepath, f"stdfit_{type}_{transform}.csv"), index=False)

    pd.DataFrame(mean_ftest).to_csv(os.path.join(filepath, f"test_meanfit_{type}_{transform}.csv"), index=False)
    pd.DataFrame(std_ftest).to_csv(os.path.join(filepath, f"test_stdfit_{type}_{transform}.csv"), index=False)

    print("- WAE")
    # Apply WAE to fuse frequencies within each fold
    wae_HIs = np.empty((repnum, foldnum), dtype=object)
    wae_fall = np.empty((repnum, foldnum))
    wae_ftest = np.empty((repnum, foldnum))
    mean_wae_fall = np.empty((foldnum))
    mean_wae_ftest = np.empty((foldnum))
    std_wae_fall = np.empty((foldnum))
    std_wae_ftest = np.empty((foldnum))
    for fold in range(foldnum):
        for rep in range(repnum):
            wae_HIs[rep, fold] = wae(HIs[rep, :, fold], fold, filepath, f"WAE_{type}_{transform}_{fold}")
    for fold in range(foldnum):
        for rep in range(repnum):
            wae_fall[rep, fold] = fitness(wae_HIs[rep, fold])[0]
            wae_ftest[rep, fold] = test_fitness(wae_HIs[rep, fold][fold], np.concatenate((wae_HIs[rep, fold][:fold], wae_HIs[rep, fold][fold+1:])))[0]
        mean_wae_fall[fold] = np.mean(wae_fall[:, fold])
        std_wae_fall[fold] = np.std(wae_fall[:, fold])
        mean_wae_ftest[fold] = np.mean(wae_ftest[:, fold])
        std_wae_ftest[fold] = np.std(wae_ftest[:, fold])

    pd.DataFrame(np.stack((mean_wae_fall, std_wae_fall), axis=0)).to_csv(os.path.join(filepath, f"weighted_{type}_{transform}.csv"), index=False)
    pd.DataFrame(np.stack((mean_wae_ftest, std_wae_ftest), axis=0)).to_csv(os.path.join(filepath, f"test_weighted_{type}_{transform}.csv"), index=False)


    # Plot HI graphs
    print("- Plotting")
    rep = 1
    for freq in range(freqnum):
        for fold in range(foldnum):
            HI_graph(HIs[rep, freq, fold], filepath, f"{freqs[freq]}kHz_{type}_{transform}_{fold}", False)
    big_plot(filepath, type, transform)

#csv_dir = r"C:\Users\pablo\Downloads\VAE_Ultimate_New"
csv_dir = r"C:\Users\Jamie\Documents\Uni\Year 2\Q3+4\Project\CSV-FFT-HLB-Reduced"
eval_wae(csv_dir, "DeepSAD", "FFT")
eval_wae(csv_dir, "DeepSAD", "HLB")