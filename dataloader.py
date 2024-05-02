import itertools
import numpy as np

def dataloader(paneldata1, paneldata2, paneldata3, paneldata4):
    data = [paneldata1, paneldata2, paneldata3, paneldata4]
    comblist = []

    for i in range(len(data) - 1):
        combs = list(itertools.combinations(data, r=(i + 1)))
        # print(combs)
        if i == 1:
            comb2 = combs
        if i == 0:
            comb4 = combs
        if i == 2:
            comb3 = combs
        comblist.append(combs)
        # print("\n")
    # print("Comblist:", comblist)

    return comblist, comb2, comb3, comb4


data1 = np.array([[2, 6, 7], [8, 1, 3], [4, 9, 5]])

data2 = np.array([[10, 2, 3], [4, 6, 7], [8, 1, 5]])

data3 = np.array([[3, 5, 9], [7, 2, 6], [1, 4, 8]])

data4 = np.array([[6, 9, 2], [1, 7, 8], [5, 4, 3]])

data = [data1, data2, data3, data4]

comblist, comb2, comb3, comb4 = dataloader(data1, data2, data3, data4)
# comb2: all comb with 2 elements
# comb3: all comb with 3 elements (123,124,134,234)
# comb4: all comb with 4 elements (1234)


# dataset1 = torch.utils.data.TensorDataset(X)
# dataset2 = torch.utils.data.TensorDataset(X)
# dataset3 = torch.utils.data.TensorDataset(X)
# dataset4 = torch.utils.data.TensorDataset(X)

# dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
