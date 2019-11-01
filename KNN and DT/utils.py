import numpy as np
from knn import KNN


############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################


# TODO: implement F1 score
def f1_score(real_labels, predicted_labels):
    tp = 0
    fp = 0
    fn = 0
    for i, j in zip(real_labels, predicted_labels):
        if i == 1 and j == 1:
            tp += 1
        if i == 1 and j == 0:
            fn += 1
        if i == 0 and j == 1:
            fp += 1
    if 2 * tp + fp + fn == 0:
        return 0
    f1_scores = float(2 * tp / (2 * tp + fp + fn))
    return f1_scores


class Distances:
    @staticmethod
    # TODO
    def minkowski_distance(point1, point2):
        sum1 = 0
        for i, j in zip(point1, point2):
            if i - j < 0:
                sum1 += (j - i) ** 3
            else:
                sum1 += (i - j) ** 3
        return sum1 ** (1 / 3)

    @staticmethod
    # TODO
    def euclidean_distance(point1, point2):
        sum2 = 0
        for i, j in zip(point1, point2):
            sum2 += (i - j) ** 2
        return sum2 ** (1 / 2)

    @staticmethod
    # TODO
    def inner_product_distance(point1, point2):
        sum3 = 0
        for i, j in zip(point1, point2):
            sum3 += i * j
        return sum3

    @staticmethod
    # TODO
    def cosine_similarity_distance(point1, point2):
        point1_len = 0
        point2_len = 0
        sum4 = 0
        for i in point1:
            point1_len += i ** 2
        for j in point2:
            point2_len += j ** 2
        for m, n in zip(point1, point2):
            sum4 += m * n
        return 1 - sum4 / (point1_len * point2_len) ** (1 / 2)

    @staticmethod
    # TODO
    def gaussian_kernel_distance(point1, point2):
        return -1 * np.exp((np.dot(np.subtract(point1, point2), np.subtract(point1, point2)) / -2))


class HyperparameterTuner:
    def __init__(self):
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None

    # TODO: find parameters with the best f1 score on validation dataset
    def tuning_without_scaling(self, distance_funcs, x_train, y_train, x_val, y_val):
        optimal_k = 0
        function = ""
        model = None
        f1_scores = -2 ** 32
        if len(x_train) <= 30:
            max_k = len(x_train)
        else:
            max_k = 30
        for key, value in distance_funcs.items():
            for k_value in range(1, max_k, 2):
                train_model = KNN(k_value, value)
                train_model.train(x_train, y_train)
                pre_val = train_model.predict(x_val)
                cur_f1 = f1_score(y_val, pre_val)
                if f1_scores < cur_f1:
                    optimal_k = k_value
                    function = key
                    model = train_model
                    f1_scores = cur_f1
        self.best_k = optimal_k
        self.best_distance_function = function
        self.best_model = model
        return self.best_k, self.best_distance_function, self.best_model

    # TODO: find parameters with the best f1 score on validation dataset, with normalized data
    def tuning_with_scaling(self, distance_funcs, scaling_classes, x_train, y_train, x_val, y_val):
        optimal_k = 0
        function = ""
        scalar = ""
        model = None
        scalar_method = []
        scalar_name = []
        f1_scores = -2 ** 32
        if len(x_train) <= 30:
            max_k = len(x_train)
        else:
            max_k = 30
        for m, n in scaling_classes.items():
            scalar_method.append(n())
            scalar_name.append(m)
        for i in range(len(scalar_method)):
            x_train = scalar_method[i](x_train)
            x_val = scalar_method[i](x_val)
            for key, value in distance_funcs.items():
                for k_value in range(1, max_k, 2):
                    train_model = KNN(k_value, value)
                    train_model.train(x_train, y_train)
                    pre_val = train_model.predict(x_val)
                    cur_f1 = f1_score(y_val, pre_val)
                    if f1_scores < cur_f1:
                        optimal_k = k_value
                        function = key
                        model = train_model
                        f1_scores = cur_f1
                        scalar = scalar_name[i]
        self.best_k = optimal_k
        self.best_distance_function = function
        self.best_scaler = scalar
        self.best_model = model
        return self.best_k, self.best_distance_function, self.best_scaler, self.best_model


class NormalizationScaler:
    def __init__(self):
        pass

    # TODO: normalize data
    def __call__(self, features):
        norm_form = []
        for i in features:
            if all(j == 0 for j in i):
                norm_form.append(i)
            else:
                mod_i = float((sum(x ** 2 for x in i)) ** (1 / 2))
                norm_feature = [k / mod_i for k in i]
                norm_form.append(norm_feature)
        return norm_form


class MinMaxScaler:
    """
    Please follow this link to know more about min max scaling
    https://en.wikipedia.org/wiki/Feature_scaling
    You should keep some states inside the object.
    You can assume that the parameter of the first __call__
    will be the training set.

    Hints:
        1. Use a variable to check for first __call__ and only compute
            and store min/max in that case.

    Note:
        1. You may assume the parameters are valid when __call__
            is being called the first time (you can find min and max).

    Example:
        train_features = [[0, 10], [2, 0]]
        test_features = [[20, 1]]

        scaler1 = MinMaxScale()
        train_features_scaled = scaler1(train_features)
        # train_features_scaled should be equal to [[0, 1], [1, 0]]

        test_features_scaled = scaler1(test_features)
        # test_features_scaled should be equal to [[10, 0.1]]

        new_scaler = MinMaxScale() # creating a new scaler
        _ = new_scaler([[1, 1], [0, 0]]) # new trainfeatures
        test_features_scaled = new_scaler(test_features)
        # now test_features_scaled should be [[20, 1]]

    """

    def __init__(self):
        self.min_val = []
        self.max_val = []

    def __call__(self, features):
        normalized_feature = features
        features_array = np.array(features)
        if len(self.min_val) == 0:
            self.min_val = np.amin(features_array, axis=0)
            self.max_val = np.amax(features_array, axis=0)
        diff = self.max_val - self.min_val
        for i in range(len(diff)):
            for j in range(len(features)):
                if diff[i] == 0:
                    normalized_feature[j][i] = self.min_val[i]
                else:
                    normalized_feature[j][i] = (features[j][i] - self.min_val[i]) / diff[i]
        return normalized_feature
