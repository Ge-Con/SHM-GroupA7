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
    newHI = np.sum(HIs/6)
    return newHI

print("Files must be 5D - 6 frequencies, n repetitions, 5 folds, 5 panels of 30 HIs")
filepath = input("Input path to file: ")
filename = input("Input HI filename:  ")

HIs = np.load(filepath + "\\" + filename)
print(HIs.shape)
if HIs.ndim != 5:
    print("Wrong number of dimensions")
else:   #Keeping folds and repetitions separate
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
    print(newHIses2)