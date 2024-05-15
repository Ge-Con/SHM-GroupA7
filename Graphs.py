import matplotlib.pyplot as plt
import numpy as np

def HI_graph(X, dir="", name=""):
    #Graph of HI against cycles
    #X is numpy list of HI at different states, & name is root to save at
    plt.figure()
    for sample in range(len(X)):
        states = np.arange(len(X[sample]))
        cycles = states*5000
        plt.plot(cycles, X[sample], label="Sample "+str(sample))
    plt.legend()
    plt.xlabel('Compression Cycles')
    plt.ylabel('HI')
    if dir != "" and name != "":
        plt.savefig(dir + "\\" + name + " HIs")
    else:
        plt.show()

def criteria_chart(features, Mo, Pr, Tr, dir="", name=""):
    #Stacked bar chart of criteria against features
    #Features is list of feature names; Mo, Pr, Tr are numpy lists of floats in same order, dir is root to save at
    plt.figure()
    plt.bar(features, Mo, label="Mo")
    plt.bar(features, Pr, bottom=Mo, label="Pr")
    plt.bar(features, Tr, bottom=Pr+Mo, label="Tr")
    plt.legend()
    plt.xlabel('Feature')
    plt.ylabel('Fitness')
    if dir != "" and name != "":
        plt.savefig(dir + "\\" + name + " PC")
    else:
        plt.show()

#criteria_chart(np.array(["f1", "f2", "f3"]), np.array([1, 2, 3]), np.array([2, 4, 6]), np.array([1, 2, 3]))
#HI_graph(np.array([1, 2, 3, 4, 5]))