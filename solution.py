import numpy as np

def get_entropy(n_cat1: int, n_cat2: int) -> float:
    total = n_cat1 + n_cat2
    p_cat1 = n_cat1 / total
    p_cat2 = n_cat2 / total

    entropy = -p_cat1 * np.log2(p_cat1) - p_cat2 * np.log2(p_cat2) if p_cat1 != 0 and p_cat2 != 0 else 0
    return entropy

def get_best_separation(features: np.ndarray, labels: np.ndarray) -> (int, int):
    num_records, num_features = features.shape
    best_info_gain = 0
    best_feature_index = 0
    best_threshold = 0

    for feature_index in range(num_features):
        feature_values = np.unique(features[:, feature_index])

        for threshold in feature_values:
            left_mask = features[:, feature_index] <= threshold
            right_mask = ~left_mask

            left_labels = labels[left_mask]
            right_labels = labels[right_mask]

            if len(left_labels) > 0 and len(right_labels) > 0:
                total_entropy = get_entropy(np.sum(labels == 0), np.sum(labels == 1))
                left_entropy = get_entropy(np.sum(left_labels == 0), np.sum(left_labels == 1))
                right_entropy = get_entropy(np.sum(right_labels == 0), np.sum(right_labels == 1))

                info_gain = total_entropy - (len(left_labels) / num_records) * left_entropy - (len(right_labels) / num_records) * right_entropy

                if info_gain > best_info_gain:
                    best_info_gain = info_gain
                    best_feature_index = feature_index
                    best_threshold = threshold

    return best_feature_index, best_threshold

class Node():
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None):
        
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        self.value = value


class DecisionTree():
    def __init__(self, min_samples_split=2):
    
        self.root = None
        self.min_samples_split = min_samples_split
        
    def build_tree(self, dataset, curr_depth=0):

        X, Y = dataset[:,:-1], dataset[:,-1]
        num_samples, num_features = np.shape(X)
    
        if num_samples>=self.min_samples_split:
            best_split = self.get_best_split(dataset, num_samples, num_features)
            if best_split["info_gain"]>0:
                left_subtree = self.build_tree(best_split["dataset_left"], curr_depth+1)
                right_subtree = self.build_tree(best_split["dataset_right"], curr_depth+1)
                return Node(best_split["feature_index"], best_split["threshold"], 
                            left_subtree, right_subtree, best_split["info_gain"])
        
        leaf_value = self.calculate_leaf_value(Y)
        return Node(value=leaf_value)
    
    def get_best_split(self, dataset, num_samples, num_features):
        ''' function to find the best split '''
        
        best_split = {}
        max_info_gain = -float("inf")
        
        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            possible_thresholds = np.unique(feature_values)
            for threshold in possible_thresholds:
                dataset_left, dataset_right = self.split(dataset, feature_index, threshold)
                if len(dataset_left)>0 and len(dataset_right)>0:
                    y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                    curr_info_gain = self.information_gain(y, left_y, right_y)
                    if curr_info_gain>max_info_gain:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["info_gain"] = curr_info_gain
                        max_info_gain = curr_info_gain
                        
        return best_split
    
    def split(self, dataset, feature_index, threshold):
        dataset_left = np.array([row for row in dataset if row[feature_index]<=threshold])
        dataset_right = np.array([row for row in dataset if row[feature_index]>threshold])
        return dataset_left, dataset_right
    
    def information_gain(self, parent, l_child, r_child):
        
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        
        gain = self.entropy(parent) - (weight_l*self.entropy(l_child) + weight_r*self.entropy(r_child))
        return gain
    
    def entropy(self, y):
        
        class_labels = np.unique(y)
        entropy = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            entropy += -p_cls * np.log2(p_cls)
        return entropy
        
    def calculate_leaf_value(self, Y):
        
        Y = list(Y)
        return max(Y, key=Y.count)
    
    def fit(self, X, Y):
        
        dataset = np.concatenate((X, Y.reshape(-1, 1)), axis=1)
        
        self.root = self.build_tree(dataset)

    def predict(self, X):
        
        preditions = [self.make_prediction(x, self.root) for x in X]
        return preditions
    
    def make_prediction(self, x, tree):
    
        if tree.value!=None: return tree.value
        feature_val = x[tree.feature_index]
        if feature_val<=tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)

def main():
    
    tree_classifier = DecisionTree(min_samples_split=2)
    train_data = np.genfromtxt('train.csv', delimiter=',')
    features_train = train_data[:, :-1]
    labels_train = train_data[:, -1]
    tree_classifier.fit(features_train, labels_train)
    test_data = np.genfromtxt('test.csv', delimiter=',')
    features_test = test_data
    predictions = tree_classifier.predict(features_test)
    with open('results.csv', 'w') as file:
        for prediction in predictions:
            file.write(f'{prediction}\n')

if __name__ == "__main__":
    main()