import numpy as np


test = np.array([[[1, 2, 3], [4, 5, 6]], [[2, 3, 4], [5, 6, 7]]])

def Signum(PC) -> float:
    sum_samples = 0
    for i in range(len(PC)):
        sum_measurements = 0
        for j in range(len(PC[i, :])-1):
                        sum_measurements += np.sign(PC[i, j+1] - PC[i, j])/((len(PC[i, :])) - 1)
        sum_samples += abs(sum_measurements)
    return sum_samples/len(PC) 

def MK(PC) -> float:
    sum_samples = 0
    for i in range(len(PC)):
        sum_measurements = 0
        for j in range(len(PC[i, :])-1):
            sub_sum = 0
            for k in range(len(PC[i, :])):
                if k > i:
                    sub_sum += (k - i) * np.sign(PC[i, k] - PC[i, j])
            sum_measurements += sub_sum
        sum_samples += abs(sum_measurements)
    return sum_samples/(len(PC) -1)

def MMK(PC) -> float:
    sum_samples = 0
    for i in range(len(PC)):
        sum_measurements = 0
        div_sum = 0
        for j in range(len(PC[i, :])-1):
            sub_sum = 0
            div_sub_sum = 0
            for k in range(len(PC[i, :])):
                if k > i:
                    sub_sum += (k - i) * np.sign(PC[i, k] - PC[i, j])
                    div_sub_sum += k-i
            sum_measurements += sub_sum
            div_sum += div_sub_sum
        sum_samples += abs(sum_measurements/div_sum)
    return sum_samples/(len(PC) -1)

np.linspace(1, 100)

line = np.fromfunction(lambda i, j: j, (5, 100))            
print(Signum(line))
