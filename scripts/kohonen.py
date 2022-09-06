import numpy as np

n1 = 16
nx = 3

rate = 0.001
change_rate = 0.01
r = 1

n_train = 120
n_test = 12

train = np.array([[0.2417, 0.2857, 0.2397],
                [0.2268, 0.2874, 0.2153],
                [0.1975, 0.3315, 0.1965],
                [0.3414, 0.3166, 0.1074],
                [0.2587, 0.1918, 0.2634],
                [0.2455, 0.2075, 0.1344],
                [0.3163, 0.1679, 0.1725],
                [0.2704, 0.2605, 0.1411],
                [0.1871, 0.2965, 0.1231],
                [0.3474, 0.2715, 0.1958],
                [0.2059, 0.2928, 0.2839],
                [0.2442, 0.2272, 0.2384],
                [0.2126, 0.3437, 0.1128],
                [0.2562, 0.2542, 0.1599],
                [0.1640, 0.2289, 0.2627],
                [0.2795, 0.1880, 0.1627],
                [0.3463, 0.1513, 0.2281],
                [0.3430, 0.1508, 0.1881],
                [0.1981, 0.2821, 0.1294],
                [0.2322, 0.3025, 0.2191],
                [0.7352, 0.2722, 0.6962],
                [0.7191, 0.1825, 0.7470],
                [0.6921, 0.1537, 0.8172],
                [0.6833, 0.2048, 0.8490],
                [0.8012, 0.2684, 0.7673],
                [0.7860, 0.1734, 0.7198],
                [0.7205, 0.1542, 0.7295],
                [0.6549, 0.3288, 0.8153],
                [0.6968, 0.3173, 0.7389],
                [0.7448, 0.2095, 0.6847],
                [0.6746, 0.3277, 0.6725],
                [0.7897, 0.2801, 0.7679],
                [0.8399, 0.3067, 0.7003],
                [0.8065, 0.3206, 0.7205],
                [0.8357, 0.3220, 0.7879],
                [0.7438, 0.3230, 0.8384],
                [0.8172, 0.3319, 0.7628],
                [0.8248, 0.2614, 0.8405],
                [0.6979, 0.2142, 0.7309],
                [0.6804, 0.3181, 0.7017],
                [0.6973, 0.3194, 0.7522],
                [0.7910, 0.2239, 0.7018],
                [0.7052, 0.2148, 0.6866],
                [0.8088, 0.1908, 0.7563],
                [0.7640, 0.1676, 0.6994],
                [0.7616, 0.2881, 0.8087],
                [0.8188, 0.2461, 0.7273],
                [0.7920, 0.3178, 0.7497],
                [0.7802, 0.1871, 0.8102],
                [0.7332, 0.2543, 0.8194],
                [0.6921, 0.1529, 0.7759],
                [0.6833, 0.2197, 0.6943],
                [0.7860, 0.1745, 0.7639],
                [0.8009, 0.3082, 0.8491],
                [0.7793, 0.1935, 0.6738],
                [0.7373, 0.2698, 0.7864],
                [0.7048, 0.2380, 0.7825],
                [0.8393, 0.2857, 0.7733],
                [0.6878, 0.2126, 0.6961],
                [0.6651, 0.3492, 0.6737],
                [0.4856, 0.6600, 0.4798],
                [0.4114, 0.7220, 0.5106],
                [0.5671, 0.7935, 0.5929],
                [0.4875, 0.7928, 0.5532],
                [0.5172, 0.7147, 0.5774],
                [0.5483, 0.6773, 0.4842],
                [0.5740, 0.6682, 0.5335],
                [0.4587, 0.6981, 0.5900],
                [0.5794, 0.7410, 0.4759],
                [0.4712, 0.6734, 0.5677],
                [0.5126, 0.8141, 0.5224],
                [0.5557, 0.7749, 0.4342],
                [0.4916, 0.8267, 0.4586],
                [0.4629, 0.8129, 0.4950],
                [0.5850, 0.7358, 0.5107],
                [0.4435, 0.7030, 0.4594],
                [0.4155, 0.7516, 0.5524],
                [0.4887, 0.7027, 0.5886],
                [0.5462, 0.7378, 0.5107],
                [0.5251, 0.8124, 0.5686],
                [0.4635, 0.7339, 0.5638],
                [0.5907, 0.7144, 0.4718],
                [0.4982, 0.8335, 0.4597],
                [0.5242, 0.7325, 0.4079],
                [0.4075, 0.8372, 0.4271],
                [0.5934, 0.8284, 0.5107],
                [0.5463, 0.6766, 0.5639],
                [0.4403, 0.8495, 0.4806],
                [0.4531, 0.7760, 0.5276],
                [0.5109, 0.7387, 0.5373],
                [0.5383, 0.7780, 0.4955],
                [0.5679, 0.7156, 0.5022],
                [0.5762, 0.7781, 0.5908],
                [0.5997, 0.7504, 0.5678],
                [0.4138, 0.6975, 0.5148],
                [0.5490, 0.6674, 0.4472],
                [0.4719, 0.7527, 0.4401],
                [0.4458, 0.8063, 0.4253],
                [0.4983, 0.8131, 0.5625],
                [0.5742, 0.6789, 0.5997],
                [0.5289, 0.7354, 0.4718],
                [0.5927, 0.7738, 0.5390],
                [0.5199, 0.7131, 0.4028],
                [0.5716, 0.6558, 0.4451],
                [0.5075, 0.7045, 0.4233],
                [0.4886, 0.7004, 0.4608],
                [0.5527, 0.8243, 0.5772],
                [0.4816, 0.6969, 0.4678],
                [0.5809, 0.6557, 0.4266],
                [0.5881, 0.7565, 0.4003],
                [0.5334, 0.8446, 0.4934],
                [0.4603, 0.7992, 0.4816],
                [0.5491, 0.6504, 0.4063],
                [0.4288, 0.8455, 0.5047],
                [0.5636, 0.7884, 0.5417],
                [0.5349, 0.6736, 0.4541],
                [0.5569, 0.8393, 0.5652],
                [0.4729, 0.7702, 0.5325],
                [0.5472, 0.8454, 0.5449],
                [0.5805, 0.7349, 0.4464]])
test = np.array([[0.2471, 0.1778, 0.2905],
                [0.8240, 0.2223, 0.7041],
                [0.4960, 0.7231, 0.5866],
                [0.2923, 0.2041, 0.2234],
                [0.8118, 0.2668, 0.7484],
                [0.4837, 0.8200, 0.4792],
                [0.3248, 0.2629, 0.2375],
                [0.7209, 0.2116, 0.7821],
                [0.5259, 0.6522, 0.5957],
                [0.2075, 0.1669, 0.1745],
                [0.7830, 0.3171, 0.7888],
                [0.5393, 0.7510, 0.5682]])

def training():
    global n_train, train, rate, n1, nx, weights, neighbours, neighbours_count,classes

    times = 0
    change = 1

    while change > change_rate:
        change = 0

        for i in range(n_train):
            distances = np.zeros(n1)

            min = 100
            min_j = -1

            for j in range(n1):
                for k in range(nx):
                    distances[j] = distances[j] + np.power(train[i, k] - weights[j, k], 2)

                distances[j] = np.sqrt(distances[j])

                if distances[j] <= min:
                    min = distances[j]
                    min_j = j

            if i < 20: classes[min_j] = 1
            elif i >= 20 and i < 60: classes[min_j] = 2
            else: classes[min_j] = 3

            for j in range(nx):
                change = change + rate * (train[i, j] - weights[min_j, j])
                weights[min_j, j] = weights[min_j, j] + rate * (train[i, j] - weights[min_j, j])

                for k in range(int(neighbours_count[min_j])):
                    change = change + (rate / 2) * (train[i, j] - weights[int(neighbours[min_j, k]), j])
                    weights[int(neighbours[min_j, k]), j] = weights[int(neighbours[min_j, k]), j] + (rate / 2) * (train[i, j] - weights[int(neighbours[min_j, k]), j])

            sum_min_j = 0
            for j in range(nx):
                sum_min_j = sum_min_j + np.power(weights[min_j, j], 2)

            sum_min_j = np.sqrt(sum_min_j)
            for j in range(nx):
                weights[min_j, j] = weights[min_j, j] / sum_min_j

            for j in range(int(neighbours_count[min_j])):
                sum = 0
                for k in range(nx):
                    sum = sum + np.power(weights[int(neighbours[min_j, j]), k], 2)

                sum = np.sqrt(sum)
                for k in range(nx):
                    weights[int(neighbours[min_j, j]), k] = weights[int(neighbours[min_j, j]), k] / sum

        times += 1

def testing():
    global n_test, test, weights, classes

    results = np.zeros(n_test)

    for i in range(n_test):
        distances = np.zeros(n1)

        min = 100
        min_j = -1

        for j in range(n1):
            for k in range(nx):
                distances[j] = distances[j] + np.power(test[i, k] - weights[j, k], 2)

            distances[j] = np.sqrt(distances[j])

            if distances[j] <= min:
                min = distances[j]
                min_j = j

        results[i] = classes[min_j]

    print("Results: ", results)

topologic_map = np.zeros((int(np.sqrt(n1)), int(np.sqrt(n1))))

n = 0
for i in range(int(np.sqrt(n1))):
    for j in range(int(np.sqrt(n1))):
        topologic_map[i, j] = n
        n += 1

weights = np.zeros((n1, nx))

for i in range(n1):
    sum = 0
    for j in range(nx):
        weights[i, j] = train[i, j]

        sum = sum + np.power(weights[i, j], 2)

    sum = np.sqrt(sum)
    for j in range(nx):
        weights[i, j] = weights[i, j] / sum

neighbours = np.zeros((n1, int(np.sqrt(n1))))
neighbours_count = np.zeros(n1)

k_mult = 0
for i in range(int(np.sqrt(n1))):
    for j in range(int(np.sqrt(n1))):
        if i - r >= 0:
            neighbours[int(topologic_map[i, j]), int(neighbours_count[k_mult + j])] = topologic_map[i - r, j]
            neighbours_count[k_mult + j] += 1
        if j - r >= 0:
            neighbours[int(topologic_map[i, j]), int(neighbours_count[k_mult + j])] = topologic_map[i, j - r]
            neighbours_count[k_mult + j] += 1
        if j + r < np.sqrt(n1):
            neighbours[int(topologic_map[i, j]), int(neighbours_count[k_mult + j])] = topologic_map[i, j + r]
            neighbours_count[k_mult + j] += 1
        if i + r < np.sqrt(n1):
            neighbours[int(topologic_map[i, j]), int(neighbours_count[k_mult + j])] = topologic_map[i + r, j]
            neighbours_count[k_mult + j] += 1

    k_mult += int(np.sqrt(n1))

classes = np.zeros(n1)

training()

print("Classes: ", classes)

testing()




