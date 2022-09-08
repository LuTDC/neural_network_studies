import numpy as np

n_train = 10
nx = 16
n1 = 16
n2 = 1

radius = 0

train = np.array([[0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1],
                    [1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0],
                    [1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1],
                    [1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0],
                    [0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1],
                    [1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1],
                    [1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0],
                    [1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1],
                    [0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1],
                    [0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1]])

def training():
    global n_train, train, weightsf, weightsb, nx, n1, n2, radius

    for i in range(n_train):
        active_neurons = np.ones(n2)

        flag = False
        tries = 0

        while not flag:
            u = np.zeros(n2)

            max = -1
            max_j = -1

            for j in range(n2):
                for k in range(nx):
                    u[j] += weightsf[j, k] * train[i, k]

                if u[j] > max and active_neurons[j] == 1:
                    max = u[j]
                    max_j = j

            r = 0

            for j in range(n1):
                r += weightsb[j, max_j] * train[i, j]

            r = r / np.sum(train[i])

            if r > radius:
                flag = True

                sum = 0
                for j in range(n1):
                    weightsb[j, max_j] = weightsb[j, max_j] * train[i, j]
                    sum += weightsb[j, max_j] * train[i, j]

                for j in range(n1):
                    weightsf[max_j, j] = (weightsb[j, max_j] * train[i, j]) / (sum + (1/2))
            else:
                flag = False
                active_neurons[max_j] = 0
                tries += 1

                if tries == n2:
                    flag = True
                    n2 += 1

                    weightsf = np.concatenate((weightsf, np.zeros((1, n1))), 0)
                    weightsb = np.concatenate((weightsb, np.zeros((n1, 1))), 1)

                    sum = 0
                    for j in range(n1):
                        weightsb[j, n2-1] = (1/(1+n1)) * train[i, j]
                        sum += weightsb[j, n2-1] * train[i, j]

                    for j in range(n1):
                        weightsf[n2-1, j] = (weightsb[j, n2-1] * train[i, j]) / (sum + (1/2))

    print("Number of classes: ", n2)

radius = 0.5
n2 = 1

weightsf = np.full((n2, n1), 1/(1+n1))
weightsb = np.ones((n1, n2))

training()

radius = 0.8
n2 = 1

weightsf = np.full((n2, n1), 1/(1+n1))
weightsb = np.ones((n1, n2))

training()

radius = 0.9
n2 = 1

weightsf = np.full((n2, n1), 1/(1+n1))
weightsb = np.ones((n1, n2))

training()

radius = 0.99
n2 = 1

weightsf = np.full((n2, n1), 1/(1+n1))
weightsb = np.ones((n1, n2))

training()

