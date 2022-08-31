import numpy as np
import matplotlib.pyplot as plt

n_train = 35
n_test = 15

nx = 4

train = np.array([[0.4329, -1.3719, 0.7022, -0.8535, 1],
                  [0.3024, 0.2286, 0.8630, 2.7909, -1],
                  [0.1349, -0.6445, 1.0530, 0.5687, -1],
                  [0.3374, -1.7163, 0.3670, -0.6283, -1],
                  [1.1434, -0.0485, 0.6637, 1.2606, 1],
                  [1.3749, -0.5071, 0.4464, 1.3009, 1],
                  [0.7221, -0.7587, 0.7681, -0.5592, 1],
                  [0.4403, -0.8072, 0.5154, -0.3129, 1],
                  [-0.5231, 0.3548, 0.2538, 1.5776, -1],
                  [0.3255, -2, 0.7112, -1.1209, 1],
                  [0.5824, 1.3915, -0.2291, 4.1735, -1],
                  [0.1340, 0.6081, 0.4450, 3.2230, -1],
                  [0.1480, -0.2988, 0.4778, 0.8649, 1],
                  [0.7359, 0.1869, -0.0872, 2.3584, 1],
                  [0.7115, -1.1469, 0.3394, 0.9573, -1],
                  [0.8251, -1.2840, 0.8452, 1.2382, -1],
                  [0.1569, 0.3712, 0.8825, 1.7633, 1],
                  [0.0033, 0.6835, 0.5389, 2.8249, -1],
                  [0.4243, 0.8313, 0.2634, 3.5855, -1],
                  [1.0490, 0.1326, 0.9138, 1.9792, 1],
                  [1.4276, 0.5331, -0.0145, 3.7286, 1],
                  [0.5971, 1.4865, 0.2904, 4.6069, -1],
                  [0.8475, 2.1479, 0.3179, 5.8235, -1],
                  [1.3967, -0.4171, 0.6443, 1.3927, 1],
                  [0.0044, 1.5378, 0.6099, 4.7755, -1],
                  [0.2201, -0.5668, 0.0515, 0.7829, 1],
                  [0.6300, -1.2480, 0.8591, 0.8093, -1],
                  [-0.2479, 0.8960, 0.0547, 1.7381, 1],
                  [-0.3088, -0.0929, 0.8659, 1.5483, -1],
                  [-0.5180, 1.4974, 0.5453, 2.3993, 1],
                  [0.6833, 0.8266, 0.0829, 2.8864, 1],
                  [0.4353, -1.4066, 0.4207, -0.4879, 1],
                  [-0.1069, -3.2329, 0.1856, -2.4572, -1],
                  [0.4662, 0.6261, 0.7304, 3.4370, -1],
                  [0.8298, -1.4089, 0.3119, 1.3235, -1]])
test = np.array([[0.9694, 0.6909, 0.4334, 3.4965],
                 [0.5427, 1.3832, 0.6390, 4.0352],
                 [0.6081, -0.9196, 0.5925, 0.1016],
                 [-0.1618, 0.4694, 0.2030, 3.0117],
                 [0.1870, -0.2578, 0.6124, 1.7749],
                 [0.4891, -0.5276, 0.4378, 0.6439],
                 [0.3777, 2.0149, 0.7423, 3.3932],
                 [1.1498, -0.4067, 0.2469, 1.5866],
                 [0.9325, 1.0950, 1.0359, 3.3591],
                 [0.5060, 1.3317, 0.9222, 3.7174],
                 [0.0497, -2.0656, 0.6124, -0.6585],
                 [0.4004, 3.5369, 0.9766, 5.3532],
                 [-0.1874, 1.3343, 0.5374, 3.2189],
                 [0.5060, 1.3317, 0.9222, 3.7174],
                 [1.6375, -0.7911, 0.7537, 0.5515]])
y = np.zeros(n_test)

rate = 0.0025
e = 0.000001

def eqm(weights):
    global n_train, train, nx

    result = 0

    for i in range(n_train):

        sum = 0
        for j in range(nx + 1):
            if j == 0:
                sum = sum + (-1) * weights[j]
            else:
                sum = sum + train[i, j - 1] * weights[j]

        result = result + np.power((train[i, 4] - sum), 2)

    result = result/n_train

    return result

def training(weights, eqm_list):
    global e, train, rate, n_train, times, nx

    previous_eqm = eqm(weights)
    current_eqm = 0

    while np.absolute(current_eqm - previous_eqm) > e:
        previous_eqm = eqm(weights)
        eqm_list.append(previous_eqm)

        for i in range(n_train):

            sum = 0
            for j in range(nx + 1):
                if j == 0:
                    sum = sum + (-1) * weights[j]
                else:
                    sum = sum + train[i, j - 1] * weights[j]

            for j in range(5):
                if j == 0:
                    weights[j] = weights[j] + rate * (train[i, 4] - sum) * (-1)
                else:
                    weights[j] = weights[j] + rate * (train[i, 4] - sum) * train[i, j - 1]

        times = times + 1
        current_eqm = eqm(weights)

def testing(weights):
    global test, y, nx

    for i in range(n_test):

        sum = 0
        for j in range(nx + 1):
            if j == 0:
                sum = sum + (-1) * weights[j]
            else:
                sum = sum + test[i, j - 1] * weights[j]

        if sum >= 0:
            y[i] = 1
        else:
            y[i] = -1

for t in range(5):
    print("T", t+1)
    print("\n")

    times = 0

    eqm_list = []
    times_list = []

    weights = np.random.uniform(0, 1, nx + 1)
    print("Original weights: ", weights)
    print("\n")

    training(weights, eqm_list)

    print("Final weights: ", weights)
    print("\n")
    print("Times: ", times - 1)
    print("\n")

    testing(weights)

    print("Results: ", y)
    print("\n")
    print("\n")

    for i in range(times):
        times_list.append(i)

    plt.plot(times_list, eqm_list)
    plt.xlabel('Times')
    plt.ylabel('EQM')
    plt.show()