import numpy as np

class DecisionTree_Node():
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
        
    def build(self, dataset, curr_depth=0):

        X, Y = dataset[:,:-1], dataset[:,-1]
        num_samples, num_features = np.shape(X)
    
        if num_samples>=self.min_samples_split:
            best_split = self.get_best_split(dataset,num_features)
            if best_split["info_gain"]>0:
                sub_left = self.build(best_split["dataset_left"], curr_depth+1)
                sub_right = self.build(best_split["dataset_right"], curr_depth+1)
                return DecisionTree_Node(
                    best_split["feature_index"],
                    best_split["threshold"], 
                    sub_left,
                    sub_right,
                    best_split["info_gain"])
        
        leaf_value = self.calculate_leaf_value(Y)
        return DecisionTree_Node(value=leaf_value)
    
    def get_best_split(self, dataset,num_features):
        
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
    

    def information_gain(self, parent, child_left, child_right):
        
        weight_left = len(child_left) / len(parent)
        weight_right = len(child_right) / len(parent)
        
        gain = self.entropy(parent) - (weight_left * self.entropy(child_left) + weight_right * self.entropy(child_right))
        return gain
    
    def predict(self, X):
        
        preditions = [self.predict_in_node(x, self.root) for x in X]
        return preditions
    
    def predict_in_node(self, x, tree):
    
        if tree.value!=None: return tree.value
        feature_val = x[tree.feature_index]
        if feature_val<=tree.threshold:
            return self.predict_in_node(x, tree.left)
        else:
            return self.predict_in_node(x, tree.right)

    def fit_dataset(self, X, Y):
        
        dataset = np.concatenate((X, Y.reshape(-1, 1)), axis=1)
        
        self.root = self.build(dataset)


def main():
    
    tree = DecisionTree(min_samples_split=2)
    train_data = np.genfromtxt('train.csv', delimiter=',')
    features_train = train_data[:, :-1]
    labels_train = train_data[:, -1]
    tree.fit_dataset(features_train, labels_train)
    test_data = np.genfromtxt('test.csv', delimiter=',')
    features_test = test_data
    with open('results.csv', 'w') as file:
        for prediction in tree.predict(features_test):
            file.write(f'{prediction}\n')

if __name__ == "__main__":
    main()