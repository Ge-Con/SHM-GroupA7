import numpy as np
import csv
from prognosticcriteria_v2 import fitness, Mo_single, Tr, Pr_single
import pandas as pd
import os

def test_fitness(test_HI, X):
    test_HI = test_HI
    monotonicity = Mo_single(test_HI)
    trendability = Tr(np.vstack([test_HI, X]))
    prognosability = Pr_single(test_HI, X)
    fitness_test = (monotonicity + trendability + prognosability), monotonicity, trendability , prognosability

    return fitness_test[0]

def wae_fitness(HIs, num):
    #Array of 6 HIs
    w = np.zeros((num))
    for run in range(num):
        w[run] = fitness(HIs[run])[0]
    w = w/np.sum(w)
    newHIs = np.zeros((5, 30))
    for run in range(num):
        newHIs += w[run] * HIs[run]
    return newHIs

def sae_fitness(HIs):
    newHI = np.sum(HIs)/HIs.shape[0]
    return newHI

print("Files must be 5D - 6 frequencies, n repetitions, 5 folds, 5 panels of 30 HIs")
filepath = "C:\\Users\Jamie\Documents\\Uni\Year 2\Q3+4\Project\sosad"
#filename = "PCA.npy"
#HIs = np.load(filepath + "\\" + filename)

if False: #HIs.ndim != 5:
    print("Wrong number of dimensions")
else:
    if True:
        files = ["42", "110", "120"]#, "130", "140"]
        #Simple averaging repetitions   #VAE - 5 folds of 6, 5, 30
        storeHIs = np.empty((3), dtype=object)
        for repetition in range(3):
            print("Repetition " + str(repetition+1))
            filename = "DF" + files[repetition] + ".npy"
            storeHIs[repetition] = np.load(filepath + "\\" + filename, allow_pickle=True)
            #print(storeHIs[repetition].shape)
        #storeHIs = np.array(storeHIs)   #3, 6, 5, 5, 30

        meanHIs = np.mean(storeHIs, axis=0)
        #stdHIs = np.std(storeHIs, axis=0)

        meanfit = np.empty((6, 5))
        stdfit = np.empty((6, 5))
        for fold in range(5):
            for freq in range(6):
                meanfit[freq, fold] = fitness(meanHIs[freq][fold])[0]
                #stdfit[freq, fold] = meanfit[freq][fold] - fitness(meanHIs[freq][fold]+stdHIs[freq][fold])[0]
        print(meanfit)
        #print(stdfit)
        pd.DataFrame(meanfit).to_csv(os.path.join(filepath, "meanfitFFT.csv"), index=False)
        #pd.DataFrame(meanfit).to_csv(os.path.join(dir, "stdfit.csv"), index=False)

        print("F-test:")
        tmeanfit = np.empty((6, 5))
        #tstdfit = np.empty((5, 6))
        for fold in range(5):
            for freq in range(6):
                tmeanfit[freq, fold] = test_fitness(meanHIs[freq][fold][fold], meanHIs[freq][fold])
                #tstdfit[freq, fold] = tmeanfit[freq, fold] - test_fitness(meanHIs[freq, fold, fold] + stdHIs[freq, fold, fold], meanHIs[freq, fold] + stdHIs[freq, fold])[0]
        print(tmeanfit)
        #print(tstdfit)
        pd.DataFrame(tmeanfit).to_csv(os.path.join(filepath, "tmeanfitFFT.csv"), index=False)
        #pd.DataFrame(meanfit).to_csv(os.path.join(dir, "tstdfit.csv"), index=False)

        newHIses = []
        for fold in range(5):
            print("-> Fold " + str(fold))
            newHIs = wae_fitness(meanHIs[:][fold], 4)  # Average repetitions
            newHIses.append(fitness(newHIs)[0])
        print(newHIses)
        pd.DataFrame(newHIses).to_csv(os.path.join(filepath, "weightedFFT.csv"), index=False)

        newHIses = []
        for fold in range(5):
            print("-> Fold " + str(fold))
            newHIs = wae_fitness(meanHIs[:][fold], 4)  # Average repetitions
            newHIses.append(test_fitness(newHIs[fold], newHIs))
        print(newHIses)
        pd.DataFrame(newHIses).to_csv(os.path.join(filepath, "tweightedFFT.csv"), index=False)

    """else:
    #Keeping folds and repetitions separate
        newHIses = []
        tHIs = []
        cols = []
        tcols = []
        for repetition in range(HIs.shape[1]):
            print("Repetition " + str(repetition))

            for fold in range(5):
                print("-> Fold " + str(fold))
                newHIs = wae_fitness(HIs[:, repetition, fold], 4) #Average repetitions
                freqs = [50, 100, 125, 150, 200, 250]
                ff = []
                fft = []
                for freq in range(6):
                    ff.append(fitness(HIs[freq, repetition, fold])[0])
                    fft.append(test_fitness(HIs[freq, repetition, fold, fold], HIs[freq, repetition, fold]))
                cols.append(ff)
                tcols.append(fft)
                tHIs.append(test_fitness(newHIs[fold], newHIs))
                newHIses.append(fitness(newHIs)[0])
        cols=np.array(cols).T
        tcols = np.array(tcols).T
        tHIs = np.array(tHIs).T
        HIs = np.array(HIs).T
        pd.DataFrame(tcols).to_csv(os.path.join(filepath, "tPCA.csv"), index=False)
        pd.DataFrame(cols).to_csv(os.path.join(filepath, "PCA.csv"), index=False)
        pd.DataFrame(tHIs).to_csv(os.path.join(filepath, "All tPCA.csv"), index=False)
        pd.DataFrame(newHIses).to_csv(os.path.join(filepath, "All PCA.csv"), index=False)"""