import numpy as np
import csv
from prognosticcriteria_v2 import fitness

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
filepath = "C:\\Users\Jamie\Documents\\Uni\Year 2\Q3+4\Project\VAE\FFT"

if False: #HIs.ndim != 5:
    print("Wrong number of dimensions")
else:
    files = ["42", "110", "120", "130", "140"]
    #Simple averaging repetitions   #VAE - 5 folds of 6, 5, 30
    storeHIs = []
    for repetition in range(5):
        print("Repetition " + str(repetition+1))
        filename = "V" + files[repetition] + ".npy"
        storeHIs.append(np.load(filepath + "\\" + filename))
        #print(storeHIs[repetition].shape)
    storeHIs = np.array(storeHIs)   #5, 5, 6, 5, 30
    #print(storeHIs)

    meanHIs = np.mean(storeHIs, axis=0)
    stdHIs = np.std(storeHIs, axis=0)

    meanfit = np.empty((5, 6))
    stdfit = np.empty((5, 6))
    for fold in range(5):
        for freq in range(6):
            meanfit[fold, freq] = fitness(meanHIs[fold, freq])[0]
            stdfit[fold, freq] = meanfit[fold, freq] - fitness(meanHIs[fold, freq]+stdHIs[fold, freq])[0]
    print(meanfit)
    print(stdfit)

    newHIses = []
    for fold in range(5):
        print("-> Fold " + str(fold))
        newHIs = wae_fitness(meanHIs[:, fold], 4)  # Average repetitions
        print(fitness(newHIs)[0])
        newHIses.append(fitness(newHIs)[0])
    print(newHIses)

    """"#Keeping folds and repetitions separate
        newHIses2 = []
        for repetition in range(HIs.shape[1]):
            print("Repetition " + str(repetition))
            newHIses = []
            for fold in range(5):
                print("-> Fold " + str(fold))
                newHIs = wae_fitness(HIs[:, repetition, fold], 4) #Average repetitions
                print(fitness(newHIs)[0])
                newHIses.append(fitness(newHIs)[0])
            newHIses2.append(newHIses)
        print(newHIses2)"""