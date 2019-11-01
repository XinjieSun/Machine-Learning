import numpy as np
from collections import Counter


class KNN:
    def __init__(self, k, distance_function):
        """
        :param k: int
        :param distance_function
        """
        self.k = k
        self.distance_function = distance_function

    # TODO: save features and lable to self
    def train(self, features, labels):
        self.train_features = features
        self.train_labels = labels
        return


    # TODO: predict labels of a list of points
    def predict(self, features):
        pre_labels = list()
        for i in features:
            all_neighbors = self.get_k_neighbors(i)
            val = {}
            for j in all_neighbors:
                if j not in val.keys():
                    val[j] = 1
                else:
                    val[j] += 1
            pre_labels.append(max(val, key=val.get))
        return list(map(int, pre_labels))


    # TODO: find KNN of one point
    def get_k_neighbors(self, point):
        distance = []
        for i in self.train_features:
            distance.append(self.distance_function(point, i))
        sorted_dis = np.argsort(distance)
        knn_dis = sorted_dis[:self.k]
        labels = []
        for k in knn_dis:
            labels.append(self.train_labels[k])
        return list(map(int, labels))



if __name__ == '__main__':
    print(np.__version__)
