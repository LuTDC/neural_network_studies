import numpy as np

n1 = 4
nx = 6

rate = 0.05

n_train = 16
n_test = 8

change_rate = 0.01

train = np.array([[2.3976, 1.5328, 1.9044, 1.1937, 2.4184, 1.8649, 1.0000],
                [2.3936, 1.4804, 1.9907, 1.2732, 2.2719, 1.8110, 1.0000],
                [2.2880, 1.4585, 1.9867, 1.2451, 2.3389, 1.8099, 1.0000],
                [2.2904, 1.4766, 1.8876, 1.2706, 2.2966, 1.7744, 1.0000],
                [1.1201, 0.0587, 1.3154, 5.3783, 3.1849, 2.4276, 2.0000],
                [0.9913, 0.1524, 1.2700, 5.3808, 3.0714, 2.3331, 2.0000],
                [1.0915, 0.1881, 1.1387, 5.3701, 3.2561, 2.3383, 2.0000],
                [1.0535, 0.1229, 1.2743, 5.3226, 3.0950, 2.3193, 2.0000],
                [1.4871, 2.3448, 0.9918, 2.3160, 1.6783, 5.0850, 3.0000],
                [1.3312, 2.2553, 0.9618, 2.4702, 1.7272, 5.0645, 3.0000],
                [1.3646, 2.2945, 1.0562, 2.4763, 1.8051, 5.1470, 3.0000],
                [1.4392, 2.2296, 1.1278, 2.4230, 1.7259, 5.0876, 3.0000],
                [2.9364, 1.5233, 4.6109, 1.3160, 4.2700, 6.8749, 4.0000],
                [2.9034, 1.4640, 4.6061, 1.4598, 4.2912, 6.9142, 4.0000],
                [3.0181, 1.4918, 4.7051, 1.3521, 4.2623, 6.7966, 4.0000],
                [2.9374, 1.4896, 4.7219, 1.3977, 4.1863, 6.8336, 4.0000]])
test = np.array([[2.9817, 1.5656, 4.8391, 1.4311, 4.1916, 6.9718],
                [1.5537, 2.2615, 1.3169, 2.5873, 1.7570, 5.0958],
                [1.2240, 0.2445, 1.3595, 5.4192, 3.2027, 2.5675],
                [2.5828, 1.5146, 2.1119, 1.2859, 2.3414, 1.8695],
                [2.4168, 1.4857, 1.8959, 1.3013, 2.4500, 1.7868],
                [1.0604, 0.2276, 1.2806, 5.4732, 3.2133, 2.4839],
                [1.5246, 2.4254, 1.1353, 2.5325, 1.7569, 5.2640],
                [3.0565, 1.6259, 4.7743, 1.3654, 4.2904, 6.9808]])

weights = np.zeros((n1, nx))

for i in range(n1):
    for j in range(n_train):
        if train[j, nx] == i + 1:
            sum = 0
            for k in range(nx):
                weights[i, k] = train[j, k]
                sum += np.power(weights[i, k], 2)

            sum = np.sqrt(sum)

            for k in range(nx):
                weights[i, k] = weights[i, k] / sum

            break

for i in range(n_train):
    sum = 0
    for j in range(nx):
        sum += np.power(train[i, j], 2)

    sum = np.sqrt(sum)
    for j in range(nx):
        train[i, j] = train[i, j] / sum

for i in range(n_test):
    sum = 0
    for j in range(nx):
        sum += np.power(test[i, j], 2)

    sum = np.sqrt(sum)
    for j in range(nx):
        test[i, j] = test[i, j] / sum

def training():
    global n_train, train, rate, weights, n1, nx

    times = 0
    change = 100

    while change > change_rate:
        change = 0

        for i in range(n_train):
            distances = np.zeros(n1)

            min = 100
            min_j = -1

            for j in range(n1):
                for k in range(nx):
                    distances[j] += np.power(train[i, k] - weights[j, k], 2)

                distances[j] = np.sqrt(distances[j])

                if distances[j] <= min:
                    min = distances[j]
                    min_j = j

            if train[i, nx] == min_j + 1:
                sum = 0
                for j in range(nx):
                    change += rate * (train[i, j] - weights[min_j, j])
                    weights[min_j, j] = weights[min_j, j] + rate * (train[i, j] - weights[min_j, j])

                    sum += np.power(weights[min_j, j], 2)

                sum = np.sqrt(sum)
                for j in range(nx):
                    weights[min_j, j] = weights[min_j, j] / sum
            else:
                sum = 0
                for j in range(nx):
                    change += rate * (train[i, j] - weights[min_j, j])
                    weights[min_j, j] = weights[min_j, j] - rate * (train[i, j] - weights[min_j, j])

                    sum += np.power(weights[min_j, j], 2)

                sum = np.sqrt(sum)
                for j in range(nx):
                    weights[min_j, j] = weights[min_j, j] / sum

        times += 1

def testing():
    global n_test, test, rate, weights, n1, nx

    results = np.zeros(n_test)

    for i in range(n_test):
        distances = np.zeros(n1)

        min = 100
        min_j = -1

        for j in range(n1):
            for k in range(nx):
                distances[j] += np.power(test[i, k] - weights[j, k], 2)

            distances[j] = np.sqrt(distances[j])

            if distances[j] <= min:
                min = distances[j]
                min_j = j

        results[i] = min_j + 1

    print("Results: ", results)

training()
testing()
