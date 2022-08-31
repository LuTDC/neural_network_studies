import numpy as np
import matplotlib.pyplot as plt

n_test = 10
n_train = 40

nx = 2
n1 = 2

rate = 0.01
e = 0.0000001

train = np.array([[0.2563, 0.9503, -1.0000],
                [0.2405, 0.9018, -1.0000],
                [0.1157, 0.3676,  1.0000],
                [0.5147, 0.0167,  1.0000],
                [0.4127, 0.3275,  1.0000],
                [0.2809, 0.5830,  1.0000],
                [0.8263, 0.9301, -1.0000],
                [0.9359, 0.8724, -1.0000],
                [0.1096, 0.9165, -1.0000],
                [0.5158, 0.8545, -1.0000],
                [0.1334, 0.1362,  1.0000],
                [0.6371, 0.1439,  1.0000],
                [0.7052, 0.6277, -1.0000],
                [0.8703, 0.8666, -1.0000],
                [0.2612, 0.6109,  1.0000],
                [0.0244, 0.5279,  1.0000],
                [0.9588, 0.3672, -1.0000],
                [0.9332, 0.5499, -1.0000],
                [0.9623, 0.2961, -1.0000],
                [0.7297, 0.5776, -1.0000],
                [0.4560, 0.1871,  1.0000],
                [0.1715, 0.7713,  1.0000],
                [0.5571, 0.5485, -1.0000],
                [0.3344, 0.0259,  1.0000],
                [0.4803, 0.7635, -1.0000],
                [0.9721, 0.4850, -1.0000],
                [0.8318, 0.7844, -1.0000],
                [0.1373, 0.0292,  1.0000],
                [0.3660, 0.8581, -1.0000],
                [0.3626, 0.7302, -1.0000],
                [0.6474, 0.3324,  1.0000],
                [0.3461, 0.2398,  1.0000],
                [0.1353, 0.8120,  1.0000],
                [0.3463, 0.1017,  1.0000],
                [0.9086, 0.1947, -1.0000],
                [0.5227, 0.2321,  1.0000],
                [0.5153, 0.2041,  1.0000],
                [0.1832, 0.0661,  1.0000],
                [0.5015, 0.9812, -1.0000],
                [0.5024, 0.5274, -1.0000]])
test = np.array([[0.8705, 0.9329, -1],
                    [0.0388, 0.2703,  1],
                    [0.8236, 0.4458, -1],
                    [0.7075, 0.1502,  1],
                    [0.9587, 0.8663, -1],
                    [0.6115, 0.9365, -1],
                    [0.3534, 0.3646,  1],
                    [0.3268, 0.2766,  1],
                    [0.6129, 0.4518, -1],
                    [0.9948, 0.4962, -1]])

weights1 = np.zeros((n1, nx))

for i in range(n1):
    for j in range(nx):
        weights1[i, j] = train[i, j]

weights2 = np.random.uniform(0, 1, n1 + 1)

ohmega = np.zeros((n1, n_train))

variances = np.zeros(n1)

def eqm():
    global n_train, train, n1, nx, weights1, weights2, variances

    result = 0

    errors = np.zeros(n_train)

    for i in range(n_train):
        y1 = np.zeros(n1)
        y2 = 0

        for j in range(n1):
            for k in range(nx):
                y1[j] = y1[j] + np.power(train[i, k] - weights1[j, k], 2)

            y1[j] = np.exp(-1 * ((y1[j]) / (2 * variances[j])))

        for j in range(n1 + 1):
            if j == 0: y2 = y2 + weights2[j] * -1
            else: y2 = y2 + weights2[j] * y1[j - 1]

        #if y2 >= 0: y2 = 1
        #else: y2 = -1

        errors[i] = np.power((train[i, nx] - y2), 2)

    result = np.sum(errors) / n_train

    return result

def training():
    global train, n_train, n1, nx, weights1, weights2, ohmega, variances

    previous_ohmega = np.ones((n1, n_train))
    counter = np.zeros(n1)

    while not np.array_equal(ohmega, previous_ohmega):
        previous_ohmega = np.copy(ohmega)

        ohmega = np.zeros((n1, n_train))

        counter = np.zeros(n1)

        for i in range(n_train):
            distances = np.zeros(n1)

            min = 1000
            min_j = -1

            for j in range(n1):
                for k in range(nx):
                    distances[j] = distances[j] + np.power(train[i, k] - weights1[j, k], 2)

                distances[j] = np.sqrt(distances[j])

                if distances[j] <= min:
                    min = distances[j]
                    min_j = j

            ohmega[min_j, int(counter[min_j])] = i
            counter[min_j] += 1

        for i in range(n1):
            size = int(counter[i])

            ohmega_sum = np.zeros(nx)

            for j in range(nx):
                for k in range(size):
                    ohmega_sum[j] = ohmega_sum[j] + train[int(ohmega[i, k]), j]

                weights1[i, j] = (1/size) * ohmega_sum[j]

    for i in range(n1):
        ohmega_sum = 0
        size = int(counter[i])

        for j in range(size):
            for k in range(nx):
                ohmega_sum = ohmega_sum + np.power(train[int(ohmega[i, j]), k] - weights1[i, k], 2)

        variances[i] = (1/size) * ohmega_sum

    y1 = np.zeros((n_train, n1))

    for i in range(n_train):
        for j in range(n1):
            for k in range(nx):
                y1[i, j] = y1[i, j] + np.power(train[i, k] - weights1[j, k], 2)

            y1[i, j] = np.exp(-1 * ((y1[i, j])/(2*variances[j])))

    previous_eqm = eqm()
    current_eqm = 0

    while np.absolute(current_eqm - previous_eqm) > e:
        previous_eqm = eqm()

        for i in range(n_train):
            y2 = 0
            for j in range(n1 + 1):
                if j == 0: y2 = y2 + weights2[j] * -1
                else: y2 = y2 + weights2[j] * y1[i, j - 1]

            #if y2 >= 0: y2 = 1
            #else: y2 = -1

            correct_rate = train[i, nx] - y2

            for j in range(n1 + 1):
                if j == 0: weights2[j] = weights2[j] + rate * correct_rate * -1
                else: weights2[j] = weights2[j] + rate * correct_rate * y1[i, j - 1]

        current_eqm = eqm()

def testing():
    global test, n_test, weights1, weights2, n1, nx, right_guesses, variances

    y = np.zeros(n_test)

    for i in range(n_test):
        y1 = np.zeros(n1)

        for j in range(n1):
            for k in range(nx):
                y1[j] = y1[j] + np.power(test[i, k] - weights1[j, k], 2)

            y1[j] = np.exp(-1 * ((y1[j]) / (2 * variances[j])))

        for j in range(n1 + 1):
            if j == 0: y[i] = y[i] + weights2[j] * -1
            else: y[i] = y[i] + weights2[j] * y1[j - 1]

        print(y[i])

        if y[i] >= 0: y[i] = 1
        else: y[i] = -1

    print("Results: ", y)

    for i in range(n_test):
        if test[i, nx] == y[i]: right_guesses += 1

    right_guesses = (right_guesses/n_test) * 100

training()

print("Centers: ", weights1)
print("Variances: ", variances)
print("\n")

print(weights2)
print("\n")

right_guesses = 0

testing()

print("\n")
print("Right guesses: ", right_guesses)
