import matplotlib.pyplot as plt
import numpy as np

def HI_graph(X):
    #Graph of HI against cycles
    #X is numpy list of HI at different states
    states = np.arange(len(X))
    cycles = states*2000
    plt.plot(cycles, X)
    plt.legend()
    plt.show()
    #plt.savefig()

def criteria_chart(features, Mo, Pr, Tr):
    #Stacked bar chart of criteria against features
    #Features is list of feature names; Mo, Pr, Tr are numpy lists of floats in same order
    plt.bar(features, Mo, label="Mo")
    plt.bar(features, Pr, bottom=Mo, label="Pr")
    plt.bar(features, Tr, bottom=Pr+Mo, label="Tr")
    plt.legend()
    plt.show()

#criteria_chart(np.array(["f1", "f2", "f3"]), np.array([1, 2, 3]), np.array([2, 4, 6]), np.array([1, 2, 3]))
#HI_graph(np.array([1, 2, 3, 4, 5]))