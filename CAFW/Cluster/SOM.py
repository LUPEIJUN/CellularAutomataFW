import numpy as np

class SOM(object):
    def __init__(self, X, output, iteration, batch_size):
        self.X = X
        self.output = output
        self.iteration = iteration
        self.batch_size = batch_size
        self.W = np.random.rand(X.shape[1], output[0] * output[1])

    def get_n(self, t):
        a = min(self.output)
        return int(a - float(a) * t / self.iteration)

    def get_eta(self, t, n):
        return np.exp(-n) / (t + 2)

    def updata_W(self, X, t, winner):
        N = self.get_n(t)
        for x, i in enumerate(winner):
            to_update = self.get_neighbors(i, N)
            for j in range(N + 1):
                eta = self.get_eta(t, j)
                for w in to_update[j]:
                    self.W[:, w] += eta * (X[x, :] - self.W[:, w])

    def get_neighbors(self, index, N):
        a, b = self.output
        length = a * b

        def distance(index1, index2):
            i1_a, i1_b = index1 // a, index1 % b
            i2_a, i2_b = index2 // a, index2 % b
            return abs(i1_a - i2_a), abs(i1_b - i2_b)

        neighbors = [set() for _ in range(N + 1)]
        for i in range(length):
            dist_a, dist_b = distance(i, index)
            if dist_a <= N and dist_b <= N:
                neighbors[max(dist_a, dist_b)].add(i)
        return neighbors

    def train(self):
        count = 0
        while count < self.iteration:
            train_X = self.X[np.random.choice(self.X.shape[0], self.batch_size)]
            self.normalize_w(self.W)
            self.normalize_x(train_X)
            train_Y = train_X.dot(self.W)
            winner = np.argmax(train_Y, axis=1)
            self.update_w(train_X, count, winner)
            count += 1
        return self.W

    def train_result(self):
        self.normalize_x(self.X)
        train_Y = self.X.dot(self.W)
        return np.argmax(train_Y, axis=1)
    
    @staticmethod
    def normalize_x(X):
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        return X
    
    @staticmethod
    def normalize_w(W):
        W /= np.linalg.norm(W, axis=0, keepdims=True)
        return W

def SOM_model(im_data_land2000, file_allfactor_2010, im_height, im_width, output_path):
    cluster_data = np.loadtxt(file_allfactor_2010)
    som_no_binary = SOM(cluster_data, (2, 3), 16, 64)
    som_no_binary.train()
    res_no_binary = som_no_binary.train_result()

    valid_pixels = [
        (row, col)
        for row in range(12, im_height - 12)
        for col in range(12, im_width - 12)
        if im_data_land2000[row][col] != 0
    ]

    data_new_no_binary = np.full((im_height, im_width), 6)

    for index, (row, col) in enumerate(valid_pixels):
        data_new_no_binary[row][col] = res_no_binary[index]

    np.savetxt(output_path, data_new_no_binary, fmt='%s', newline='\n')