import numpy as np

n_train = 30
n_test = 10

train = np.array([[-0.6508, 0.1097, 4.0009, -1],
                  [-1.4492, 0.8896, 4.4005, -1],
                  [2.0850, 0.6876, 12.0710, -1],
                  [0.2626, 1.1476, 7.7985, 1],
                  [0.6418, 1.0234, 7.0427, 1],
                  [0.2569, 0.6730, 8.3265, -1],
                  [1.1155, 0.6043, 7.4446, 1],
                  [0.0914, 0.3399, 7.0677, -1],
                  [0.0121, 0.5256, 4.6316, 1],
                  [-0.0429, 0.4660, 5.4323, 1],
                  [0.4340, 0.6870, 8.2287, -1],
                  [0.2735, 1.0287, 7.1934, 1],
                  [0.4839, 0.4851, 7.4850, -1],
                  [0.4089, -0.1267, 5.5019, -1],
                  [1.4391, 0.1614, 8.5843, -1],
                  [-0.9115, -0.1973, 2.1962, -1],
                  [0.3654, 1.0475, 7.4858, 1],
                  [0.2144, 0.7515, 7.1699, 1],
                  [0.2013, 1.0014, 6.5489, 1],
                  [0.6483, 0.2183, 5.8991, 1],
                  [-0.1147, 0.2242, 7.2435, -1],
                  [-0.7970, 0.8795, 3.8762, 1],
                  [-1.0625, 0.6366, 2.4707, 1],
                  [0.5307, 0.1285, 5.6883, 1],
                  [-1.2200, 0.7777, 1.7252, 1],
                  [0.3957, 0.1076, 5.6623, -1],
                  [-0.1013, 0.5989, 7.1812, -1],
                  [2.4482, 0.9455, 11.2095, 1],
                  [2.0149, 0.6192, 10.9263, -1],
                  [0.2012, 0.2611, 5.4631, 1]])
test = np.array([[-0.3665, 0.0620, 5.9891],
        [-0.7842, 1.1267, 5.5912],
        [0.3012, 0.5611, 5.8234],
        [0.7757, 1.0648, 8.0677],
        [0.1570, 0.8028, 6.3040],
        [-0.7014, 1.0316, 3.6005],
        [0.3748, 0.1536, 6.1537],
        [-0.6920, 0.9404, 4.4058],
        [-1.3970, 0.7141, 4.9263],
        [-1.8842, -0.2805, 1.2548]])
y = np.zeros(n_test)

rate = 0.3
times = 0

def training(weights):
    global train, rate, n_train, times
    flag = False

    while not flag:
        flag = True

        for i in range(n_train):
            sum = 0
            for j in range(4):
                if j == 0:
                    sum = sum + (-1) * weights[j]
                else:
                    sum = sum + train[i, j - 1] * weights[j]

            if sum >= 0: sum = 1
            else: sum = -1

            if sum != train[i, 3]:
                flag = False

                for j in range(4):
                    if j == 0:
                        weights[j] = weights[j] + rate * (train[i, 3] - sum) * -1
                    else:
                        weights[j] = weights[j] + rate * (train[i, 3] - sum) * train[i, j - 1]

        times = times + 1

def testing(weights):
    global test, y

    sum = 0
    for i in range(n_test):
        for j in range(4):
            if j == 0:
                sum = sum + (-1) * weights[j]
            else:
                sum = sum + test[i, j - 1] * weights[j]

        if sum >= 0: y[i] = 1
        else: y[i] = -1

for t in range(5):
    times = 0

    weights = np.random.uniform(0, 1, 4)
    print("Original weights: ", weights)

    training(weights)

    print("Final weights: ", weights)
    print("Times: ", times-1)

    testing(weights)

    print("Results: ", y)