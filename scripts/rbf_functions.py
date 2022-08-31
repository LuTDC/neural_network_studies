import numpy as np
import matplotlib.pyplot as plt

n_test = 15
n_train = 150

nx = 3

rate = 0.01
e = 0.0000001

train = np.array([[0.9532, 0.6949, 0.4451, 0.8426],
                [0.7954, 0.8346, 0.0449, 0.6676],
                [0.1427, 0.0480, 0.6267, 0.3780],
                [0.1516, 0.9824, 0.0827, 0.4627],
                [0.4868, 0.6223, 0.7462, 0.8116],
                [0.3408, 0.5115, 0.0783, 0.4559],
                [0.8146, 0.6378, 0.5837, 0.8628],
                [0.2820, 0.5409, 0.7256, 0.6939],
                [0.5716, 0.2958, 0.5477, 0.6619],
                [0.9323, 0.0229, 0.4797, 0.5731],
                [0.2907, 0.7245, 0.5165, 0.6911],
                [0.0068, 0.0545, 0.0861, 0.0851],
                [0.2636, 0.9885, 0.2175, 0.5847],
                [0.0350, 0.3653, 0.7801, 0.5117],
                [0.9670, 0.3031, 0.7127, 0.7836],
                [     0, 0.7763, 0.8735, 0.6388],
                [0.4395, 0.0501, 0.9761, 0.5712],
                [0.9359, 0.0366, 0.9514, 0.6826],
                [0.0173, 0.9548, 0.4289, 0.5527],
                [0.6112, 0.9070, 0.6286, 0.8803],
                [0.2010, 0.9573, 0.6791, 0.7283],
                [0.8914, 0.9144, 0.2641, 0.7966],
                [0.0061, 0.0802, 0.8621, 0.3711],
                [0.2212, 0.4664, 0.3821, 0.5260],
                [0.2401, 0.6964, 0.0751, 0.4637],
                [0.7881, 0.9833, 0.3038, 0.8049],
                [0.2435, 0.0794, 0.5551, 0.4223],
                [0.2752, 0.8414, 0.2797, 0.6079],
                [0.7616, 0.4698, 0.5337, 0.7809],
                [0.3395, 0.0022, 0.0087, 0.1836],
                [0.7849, 0.9981, 0.4449, 0.8641],
                [0.8312, 0.0961, 0.2129, 0.4857],
                [0.9763, 0.1102, 0.6227, 0.6667],
                [0.8597, 0.3284, 0.6932, 0.7829],
                [0.9295, 0.3275, 0.7536, 0.8016],
                [0.2435, 0.2163, 0.7625, 0.5449],
                [0.9281, 0.8356, 0.5285, 0.8991],
                [0.8313, 0.7566, 0.6192, 0.9047],
                [0.1712, 0.0545, 0.5033, 0.3561],
                [0.0609, 0.1702, 0.4306, 0.3310],
                [0.5899, 0.9408, 0.0369, 0.6245],
                [0.7858, 0.5115, 0.0916, 0.6066],
                [1.0000, 0.1653, 0.7103, 0.7172],
                [0.2007, 0.1163, 0.3431, 0.3385],
                [0.2306, 0.0330, 0.0293, 0.1590],
                [0.8477, 0.6378, 0.4623, 0.8254],
                [0.9677, 0.7895, 0.9467, 0.9782],
                [0.0339, 0.4669, 0.1526, 0.3250],
                [0.0080, 0.8988, 0.4201, 0.5404],
                [0.9955, 0.8897, 0.6175, 0.9360],
                [0.7408, 0.5351, 0.2732, 0.6949],
                [0.6843, 0.3737, 0.1562, 0.5625],
                [0.8799, 0.7998, 0.3972, 0.8399],
                [0.5700, 0.5111, 0.2418, 0.6258],
                [0.6796, 0.4117, 0.3370, 0.6622],
                [0.3567, 0.2967, 0.6037, 0.5969],
                [0.3866, 0.8390, 0.0232, 0.5316],
                [0.0271, 0.7788, 0.7445, 0.6335],
                [0.8174, 0.8422, 0.3229, 0.8068],
                [0.6027, 0.1468, 0.3759, 0.5342],
                [0.1203, 0.3260, 0.5419, 0.4768],
                [0.1325, 0.2082, 0.4934, 0.4105],
                [0.6950, 1.0000, 0.4321, 0.8404],
                [0.0036, 0.1940, 0.3274, 0.2697],
                [0.2650, 0.0161, 0.5947, 0.4125],
                [0.5849, 0.6019, 0.4376, 0.7464],
                [0.0108, 0.3538, 0.1810, 0.2800],
                [0.9008, 0.7264, 0.9184, 0.9602],
                [0.0023, 0.9659, 0.3182, 0.4986],
                [0.1366, 0.6357, 0.6967, 0.6459],
                [0.8621, 0.7353, 0.2742, 0.7718],
                [0.0682, 0.9624, 0.4211, 0.5764],
                [0.6112, 0.6014, 0.5254, 0.7868],
                [0.0030, 0.7585, 0.8928, 0.6388],
                [0.7644, 0.5964, 0.0407, 0.6055],
                [0.6441, 0.2097, 0.5847, 0.6545],
                [0.0803, 0.3799, 0.6020, 0.4991],
                [0.1908, 0.8046, 0.5402, 0.6665],
                [0.6937, 0.3967, 0.6055, 0.7595],
                [0.2591, 0.0582, 0.3978, 0.3604],
                [0.4241, 0.1850, 0.9066, 0.6298],
                [0.3332, 0.9303, 0.2475, 0.6287],
                [0.3625, 0.1592, 0.9981, 0.5948],
                [0.9259, 0.0960, 0.1645, 0.4716],
                [0.8606, 0.6779, 0.0033, 0.6242],
                [0.0838, 0.5472, 0.3758, 0.4835],
                [0.0303, 0.9191, 0.7233, 0.6491],
                [0.9293, 0.8319, 0.9664, 0.9840],
                [0.7268, 0.1440, 0.9753, 0.7096],
                [0.2888, 0.6593, 0.4078, 0.6328],
                [0.5515, 0.1364, 0.2894, 0.4745],
                [0.7683, 0.0067, 0.5546, 0.5708],
                [0.6462, 0.6761, 0.8340, 0.8933],
                [0.3694, 0.2212, 0.1233, 0.3658],
                [0.2706, 0.3222, 0.9996, 0.6310],
                [0.6282, 0.1404, 0.8474, 0.6733],
                [0.5861, 0.6693, 0.3818, 0.7433],
                [0.6057, 0.9901, 0.5141, 0.8466],
                [0.5915, 0.5588, 0.3055, 0.6787],
                [0.8359, 0.4145, 0.5016, 0.7597],
                [0.5497, 0.6319, 0.8382, 0.8521],
                [0.7072, 0.1721, 0.3812, 0.5772],
                [0.1185, 0.5084, 0.8376, 0.6211],
                [0.6365, 0.5562, 0.4965, 0.7693],
                [0.4145, 0.5797, 0.8599, 0.7878],
                [0.2575, 0.5358, 0.4028, 0.5777],
                [0.2026, 0.3300, 0.3054, 0.4261],
                [0.3385, 0.0476, 0.5941, 0.4625],
                [0.4094, 0.1726, 0.7803, 0.6015],
                [0.1261, 0.6181, 0.4927, 0.5739],
                [0.1224, 0.4662, 0.2146, 0.4007],
                [0.6793, 0.6774, 1.0000, 0.9141],
                [0.8176, 0.0358, 0.2506, 0.4707],
                [0.6937, 0.6685, 0.5075, 0.8220],
                [0.2404, 0.5411, 0.8754, 0.6980],
                [0.6553, 0.2609, 0.1188, 0.4851],
                [0.8886, 0.0288, 0.2604, 0.4802],
                [0.3974, 0.5275, 0.6457, 0.7215],
                [0.2108, 0.4910, 0.5432, 0.5913],
                [0.8675, 0.5571, 0.1849, 0.6805],
                [0.5693, 0.0242, 0.9293, 0.6033],
                [0.8439, 0.4631, 0.6345, 0.8226],
                [0.3644, 0.2948, 0.3937, 0.5240],
                [0.2014, 0.6326, 0.9782, 0.7143],
                [0.4039, 0.0645, 0.4629, 0.4547],
                [0.7137, 0.0670, 0.2359, 0.4602],
                [0.4277, 0.9555,      0, 0.5477],
                [0.0259, 0.7634, 0.2889, 0.4738],
                [0.1871, 0.7682, 0.9697, 0.7397],
                [0.3216, 0.5420, 0.0677, 0.4526],
                [0.2524, 0.7688, 0.9523, 0.7711],
                [0.3621, 0.5295, 0.2521, 0.5571],
                [0.2942, 0.1625, 0.2745, 0.3759],
                [0.8180, 0.0023, 0.1439, 0.4018],
                [0.8429, 0.1704, 0.5251, 0.6563],
                [0.9612, 0.6898, 0.6630, 0.9128],
                [0.1009, 0.4190, 0.0826, 0.3055],
                [0.7071, 0.7704, 0.8328, 0.9298],
                [0.3371, 0.7819, 0.0959, 0.5377],
                [0.9931, 0.6727, 0.3139, 0.7829],
                [0.9123,      0, 0.1106, 0.3944],
                [0.2858, 0.9688, 0.2262, 0.5988],
                [0.7931, 0.8993, 0.9028, 0.9728],
                [0.7841, 0.0778, 0.9012, 0.6832],
                [0.1380, 0.5881, 0.2367, 0.4622],
                [0.6345, 0.5165, 0.7139, 0.8191],
                [0.2453, 0.5888, 0.1559, 0.4765],
                [0.1174, 0.5436, 0.3657, 0.4953],
                [0.3667, 0.3228, 0.6952, 0.6376],
                [0.2204, 0.1785, 0.4607, 0.4276]])
test = np.array([[0.5102, 0.7464, 0.0860, 0.5965],
                [0.8401, 0.4490, 0.2719, 0.6790],
                [0.1283, 0.1882, 0.7253, 0.4662],
                [0.2299, 0.1524, 0.7353, 0.5012],
                [0.3209, 0.6229, 0.5233, 0.6810],
                [0.8203, 0.0682, 0.4260, 0.5643],
                [0.3471, 0.8889, 0.1564, 0.5875],
                [0.5762, 0.8292, 0.4116, 0.7853],
                [0.9053, 0.6245, 0.5264, 0.8506],
                [0.8149, 0.0396, 0.6227, 0.6165],
                [0.1016, 0.6382, 0.3173, 0.4957],
                [0.9108, 0.2139, 0.4641, 0.6625],
                [0.2245, 0.0971, 0.6136, 0.4402],
                [0.6423, 0.3229, 0.8567, 0.7663],
                [0.5252, 0.6529, 0.5729, 0.7893]])

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
    global train, n_train, n1, nx, weights1, weights2, ohmega, variances, times, eqm_list

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
        eqm_list.append(previous_eqm)

        for i in range(n_train):
            y2 = 0
            for j in range(n1 + 1):
                if j == 0: y2 = y2 + weights2[j] * -1
                else: y2 = y2 + weights2[j] * y1[i, j - 1]

            correct_rate = train[i, nx] - y2

            for j in range(n1 + 1):
                if j == 0: weights2[j] = weights2[j] + rate * correct_rate * -1
                else: weights2[j] = weights2[j] + rate * correct_rate * y1[i, j - 1]

        current_eqm = eqm()
        times += 1

    print("Times: ", times)
    print("EQM: ", current_eqm)

def testing():
    global test, n_test, weights1, weights2, n1, nx, variances, error, variance

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

        error = error + (((test[i, nx] - y[i]) / y[i]) * 100)

    error = error / n_test

    for i in range(n_test):
        variance = variance + np.power((y[i] - error), 2)

    variance = variance / (n_test - 1)

    print("Results: ", y)
    print("\n")

n1 = 5

for t in range(3):
    print("Topology 1 - T", t+1)
    print("\n")

    times = 0

    weights1 = np.zeros((n1, nx))

    for i in range(n1):
        for j in range(nx):
            weights1[i, j] = train[i, j]

    weights2 = np.random.uniform(0, 1, n1 + 1)

    ohmega = np.zeros((n1, n_train))

    variances = np.zeros(n1)

    final_eqm = 0

    eqm_list = []
    times_list = []

    training()

    for i in range(times):
        times_list.append(i)

    print("\n")

    error = 0
    variance = 0

    testing()

    print("Error: ", error)
    print("Variance: ", variance)
    print("\n")

    plt.plot(times_list, eqm_list)
    plt.xlabel('Times')
    plt.ylabel('EQM')
    plt.show()

n1 = 10

for t in range(3):
    print("Topology 2 - T", t + 1)
    print("\n")

    times = 0

    weights1 = np.zeros((n1, nx))

    for i in range(n1):
        for j in range(nx):
            weights1[i, j] = train[i, j]

    weights2 = np.random.uniform(0, 1, n1 + 1)

    ohmega = np.zeros((n1, n_train))

    variances = np.zeros(n1)

    final_eqm = 0

    eqm_list = []
    times_list = []

    training()

    for i in range(times):
        times_list.append(i)

    print("\n")

    error = 0
    variance = 0

    testing()

    print("Error: ", error)
    print("Variance: ", variance)
    print("\n")

    plt.plot(times_list, eqm_list)
    plt.xlabel('Times')
    plt.ylabel('EQM')
    plt.show()

n1 = 15

for t in range(3):
    print("Topology 3 - T", t + 1)
    print("\n")

    times = 0

    weights1 = np.zeros((n1, nx))

    for i in range(n1):
        for j in range(nx):
            weights1[i, j] = train[i, j]

    weights2 = np.random.uniform(0, 1, n1 + 1)

    ohmega = np.zeros((n1, n_train))

    variances = np.zeros(n1)

    final_eqm = 0

    eqm_list = []
    times_list = []

    training()

    for i in range(times):
        times_list.append(i)

    print("\n")

    error = 0
    variance = 0

    testing()

    print("Error: ", error)
    print("Variance: ", variance)
    print("\n")

    plt.plot(times_list, eqm_list)
    plt.xlabel('Times')
    plt.ylabel('EQM')
    plt.show()


