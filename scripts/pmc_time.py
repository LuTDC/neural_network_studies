import numpy as np
import matplotlib.pyplot as plt

n_train = 100
n_test = 20

n1 = 0
p = 0

train = np.array([0.1701, 0.1023, 0.4405, 0.3609, 0.7192, 0.2258, 0.3175, 0.0127, 0.4290, 0.0544, 0.8000, 0.0450, 0.4268, 0.0112, 0.3218, 0.2185, 0.7240, 0.3516, 0.4420, 0.0984, 0.1747, 0.3964, 0.5114, 0.6183, 0.3330, 0.2398, 0.0508, 0.4497, 0.2178, 0.7762, 0.1078, 0.3773, 0.0001, 0.3877, 0.0821, 0.7836, 0.1887, 0.4483, 0.0424, 0.2539, 0.3164, 0.6386, 0.4862, 0.4068, 0.1611, 0.1101, 0.4372, 0.3795, 0.7092, 0.2400, 0.3087, 0.0159, 0.4330, 0.0733, 0.7995, 0.0262, 0.4223, 0.0085, 0.3303, 0.2037, 0.7332, 0.3328, 0.4445, 0.0909, 0.1838, 0.3888, 0.5277, 0.6042, 0.3435, 0.2304, 0.0568, 0.4500, 0.2371, 0.7705, 0.1246, 0.3701, 0.0006, 0.3943, 0.0646, 0.7878, 0.1694, 0.4468, 0.0372, 0.2632, 0.3048, 0.6516, 0.4690, 0.4132, 0.1523, 0.1182, 0.4334, 0.3978, 0.6987, 0.2538, 0.2998, 0.0195, 0.4366, 0.0924, 0.7984, 0.0077])
test = np.array([0.4173, 0.0062, 0.3387, 0.1886, 0.7418, 0.3138, 0.4466, 0.0835, 0.1930, 0.3807, 0.5438, 0.5897, 0.3536, 0.2210, 0.0631, 0.4499, 0.2564, 0.7642, 0.1411, 0.3626])
y = np.zeros(0)

t_test = np.arange(101, 121)

rate = 0.01
e = 0.0000005
alpha = 0.8

def eqm(weights1, weights2):
    global n_train, train, p, n1

    result = 0

    errors = np.zeros(n_train - p)
    y2 = np.zeros(n_train)

    for i in range(p):
        y2[i] = train[i]

    for i in range(n_train - p):
        l1 = np.zeros(n1)
        l2 = 0

        y1 = np.zeros(n1)

        for j in range(n1):
            for k in range(p + 1):
                if k == 0: l1[j] = l1[j] + weights1[j, k] * -1
                else: l1[j] = l1[j] + weights1[j, k] * train[i + k - 1]

            y1[j] = 1/(1 + np.exp(-l1[j]))

        for j in range(n1 + 1):
            if j == 0: l2 = l2 + weights2[j] * -1
            else: l2 = l2 + weights2[j] * y1[j - 1]

        y2[i + p] = 1/(1 + np.exp(-l2))

        errors[i] = np.power((train[i + p] - y2[i + p]), 2)

    result = np.sum(errors) / (n_train - p)

    return result

def training(weights1, weights2, eqm_list):
    global e, train, rate, n_train, times, alpha, p, n1, final_eqm

    previous_eqm = eqm(weights1, weights2)
    current_eqm = 0

    previous_weights1 = np.copy(weights1)
    previous_weights2 = np.copy(weights2)

    while np.absolute(current_eqm - previous_eqm) > e and times <= 50000:
        previous_eqm = eqm(weights1, weights2)
        eqm_list.append(previous_eqm)

        y2 = np.zeros(n_train)

        for i in range(p):
            y2[i] = train[i]

        for i in range(n_train - p):
            l1 = np.zeros(n1)
            l2 = 0

            y1 = np.zeros(n1)

            for j in range(n1):
                for k in range(p + 1):
                    if k == 0: l1[j] = l1[j] + weights1[j, k] * -1
                    else: l1[j] = l1[j] + weights1[j, k] * train[i + k - 1]

                y1[j] = 1 / (1 + np.exp(-l1[j]))

            for j in range(n1 + 1):
                if j == 0: l2 = l2 + weights2[j] * -1
                else: l2 = l2 + weights2[j] * y1[j - 1]

            y2[i + p] = 1 / (1 + np.exp(-l2))

            correct_rate1 = np.zeros(n1)
            correct_rate2 = 0

            correct_rate2 = (train[i + p] - y2[i + p]) * (y2[i + p] * (1 - y2[i + p]))

            for j in range(n1 + 1):
                previous_weight = weights2[j]

                if j == 0: weights2[j] = weights2[j] + rate * correct_rate2 * -1 + alpha * (weights2[j] - previous_weights2[j])
                else: weights2[j] = weights2[j] + rate * correct_rate2 * y1[j - 1] + alpha * (weights2[j] - previous_weights2[j])

                previous_weights2[j] = previous_weight

            for j in range(n1):
                correct_rate1[j] = correct_rate1[j] + correct_rate2 * weights2[j + 1]

                correct_rate1[j] = correct_rate1[j] * (y1[j] * (1 - y1[j])) * -1

            for j in range(n1):
                for k in range(p + 1):
                    previous_weight = weights1[j, k]

                    if k == 0: weights1[j, k] = weights1[j, k] + rate * correct_rate1[j] * -1 + alpha * (weights1[j, k] - previous_weights1[j, k])
                    else: weights1[j, k] = weights1[j, k] + rate * correct_rate1[j] * y2[i + k - 1] + alpha * (weights1[j, k] - previous_weights1[j, k])

                    previous_weights1[j, k] = previous_weight

        times = times + 1
        current_eqm = eqm(weights1, weights2)

    final_eqm = current_eqm

def testing(weights1, weights2):
    global n_test, test, y, error, variance

    for i in range(n_test):
        l1 = np.zeros(n1)
        l2 = 0

        y1 = np.zeros(n1)

        for j in range(n1):
            for k in range(p + 1):
                if k == 0: l1[j] = l1[j] + weights1[j, k] * -1
                else: l1[j] = l1[j] + weights1[j, k] * y[i + k - 1]

            y1[j] = 1 / (1 + np.exp(-l1[j]))

        for j in range(n1 + 1):
            if j == 0: l2 = l2 + weights2[j] * -1
            else: l2 = l2 + y1[j - 1] * weights2[j]

        y[i + p] = 1 / (1 + np.exp(-l2))

        error = error + (((test[i] - y[i + p]) / y[i + p]) * 100)

    error = error/n_test

    for i in range(n_test):
        variance = variance + np.power((y[i + p] - error), 2)

    variance = variance / (n_test - 1)

n1 = 10
p = 5

y = np.zeros(n_test + p)

i_aux = p
for i in range(p):
    y[i] = train[n_train - i_aux]
    i_aux -= 1

for t in range(3):
    print("T", t+1, " - TDNN 1")
    print("\n")

    times = 0
    final_eqm = 0

    eqm_list = []
    times_list = []

    weights1 = np.random.uniform(0, 1, (n1, p + 1))
    weights2 = np.random.uniform(0, 1, n1 + 1)

    training(weights1, weights2, eqm_list)

    print("Times: ", times)
    print("EQM: ", final_eqm)
    print("\n")

    for i in range(times):
        times_list.append(i)

    #plt.plot(times_list, eqm_list)
    plt.xlabel('Times')
    plt.ylabel('EQM')
    #plt.show()

    error = 0
    variance = 0

    testing(weights1, weights2)

    #plt.plot(t_test, test)
    plt.xlabel('t')
    plt.ylabel('d')
    #plt.show()

    #plt.plot(t_test, y[p:])
    plt.xlabel('t')
    plt.ylabel('y')
    #plt.show()

    print("Results: ", y[p:])
    print("\n")
    print("Error: ", error)
    print("Variance: ", variance)
    print("\n")
    print("\n")

n1 = 15
p = 10

y = np.zeros(n_test + p)

i_aux = p
for i in range(p):
    y[i] = train[n_train - i_aux]
    i_aux -= 1

for t in range(3):
    print("T", t+1, " - TDNN 2")
    print("\n")

    times = 0
    final_eqm = 0

    eqm_list = []
    times_list = []

    weights1 = np.random.uniform(0, 1, (n1, p + 1))
    weights2 = np.random.uniform(0, 1, n1 + 1)

    training(weights1, weights2, eqm_list)

    print("Times: ", times)
    print("EQM: ", final_eqm)
    print("\n")

    for i in range(times):
        times_list.append(i)

    #plt.plot(times_list, eqm_list)
    plt.xlabel('Times')
    plt.ylabel('EQM')
    #plt.show()

    error = 0
    variance = 0

    testing(weights1, weights2)

    #plt.plot(t_test, test)
    plt.xlabel('t')
    plt.ylabel('d')
    #plt.show()

    #plt.plot(t_test, y[p:])
    plt.xlabel('t')
    plt.ylabel('y')
    #plt.show()

    print("Results: ", y[p:])
    print("\n")
    print("Error: ", error)
    print("Variance: ", variance)
    print("\n")
    print("\n")

n1 = 25
p = 15

y = np.zeros(n_test + p)

i_aux = p
for i in range(p):
    y[i] = train[n_train - i_aux]
    i_aux -= 1

for t in range(3):
    print("T", t+1, " - TDNN 3")
    print("\n")

    times = 0
    final_eqm = 0

    eqm_list = []
    times_list = []

    weights1 = np.random.uniform(0, 1, (n1, p + 1))
    weights2 = np.random.uniform(0, 1, n1 + 1)

    training(weights1, weights2, eqm_list)

    print("Times: ", times)
    print("EQM: ", final_eqm)
    print("\n")

    for i in range(times):
        times_list.append(i)

    #plt.plot(times_list, eqm_list)
    plt.xlabel('Times')
    plt.ylabel('EQM')
    #plt.show()

    error = 0
    variance = 0

    testing(weights1, weights2)

    #plt.plot(t_test, test)
    plt.xlabel('t')
    plt.ylabel('d')
    #plt.show()

    #plt.plot(t_test, y[p:])
    plt.xlabel('t')
    plt.ylabel('y')
    #plt.show()

    print("Results: ", y[p:])
    print("\n")
    print("Error: ", error)
    print("Variance: ", variance)
    print("\n")
    print("\n")