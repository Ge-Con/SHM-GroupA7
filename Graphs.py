import matplotlib.pyplot as plt
import numpy as np

def HI_graph(X, dir="", name=""):
    #Graph of HI against cycles
    #X is numpy list of HI at different states, & name is root to save at
    samples = ["PZT-FFT-HLB-L1-03", "PZT-FFT-HLB-L1-04", "PZT-FFT-HLB-L1-05", "PZT-FFT-HLB-L1-09", "PZT-FFT-HLB-L1-23"]
    plt.figure()
    for sample in range(len(X)):
        states = np.arange(len(X[sample]))
        cycles = states/30*100
        if str(sample) == str(name[-1]) or samples[sample] == name[:-7]:
            plt.plot(cycles, X[sample], label="Sample "+str(sample+1) + ": Test")
        else:
            plt.plot(cycles, X[sample], label="Sample " + str(sample+1) + ": Train")
    plt.legend()
    plt.xlabel('Lifetime (%)')
    plt.ylabel('HI')
    if dir != "" and name != "":
        plt.savefig(dir + "\\" + name)
    else:
        plt.show()
    plt.close()

def criteria_chart(features, Mo, Pr, Tr, dir="", name=""):
    #Stacked bar chart of criteria against features
    #Features is list of feature names; Mo, Pr, Tr are numpy lists of floats in same order, dir is root to save at
    plt.figure()
    plt.bar(features, Mo, label="Mo")
    plt.bar(features, Pr, bottom=Mo, label="Pr")
    plt.bar(features, Tr, bottom=Pr+Mo, label="Tr")
    plt.legend()
    if features[0] == "050":
        plt.xlabel('Frequency (kHz)')
    else:
        plt.xlabel('Feature')
    plt.ylabel('Fitness')
    if dir != "" and name != "":
        plt.savefig(dir + "\\" + name + " PC")
    else:
        plt.show()
    plt.close()

#criteria_chart(np.array(["f1", "f2", "f3"]), np.array([1, 2, 3]), np.array([2, 4, 6]), np.array([1, 2, 3]))
#HI_graph(np.array([1, 2, 3, 4, 5]))