import numpy as np
class DTLearner(object):
    def __init__(self, leaf_size=1, verbose=False):
        self.leaf_size = leaf_size
        self.verbose = verbose


    def author(self):
        return "zdang31"
    def study_group(self):
        return "zdang31" 
        
    def add_evidence(self, data_x, data_y):
        # end recursion when reaching leaf size
        if data_x.shape[0] <= self.leaf_size:
            # 2d fix shape issue. otherwise, (3,) instead of (3,1) will cause mistake for right tree calculation
            tree= np.atleast_2d(np.array(["leaf", np.mean(data_y), np.nan, np.nan],dtype=object))
        # end recursion when all values are same
        elif len(np.unique(data_y)) == 1:
            tree= np.atleast_2d(np.array(["leaf", data_y[0], np.nan, np.nan],dtype=object))
        else:
            #compare correlation between each column in data_x and data_y. handling the constant and avoiding nan
            corr = np.array([np.corrcoef(data_x[:, i], data_y)[0, 1] if np.std(data_x[:, i]) > 0 else 0 for i in range(data_x.shape[1])])
            #get max correlation. (isclose to fix index float point issue, otherwise, some feature with correlation of 0.9999 will be ignored)
            max_corr = np.where(np.isclose(np.abs(corr), np.max(np.abs(corr))))[0]
            #selet max correlation column as feature
            if len(max_corr) > 1:
                index = max_corr[0]
            else:
                index = np.argmax(np.abs(corr))
            #using median of selected column as split value.
            split_value = np.median(data_x[:, index])
            #slicing left and right tree data
            left_indices= data_x[:, index] <= split_value
            right_indices= data_x[:, index] > split_value
            left_data_x= data_x[left_indices]
            left_data_y=data_y[left_indices]
            right_data_x=data_x[right_indices]
            right_data_y=data_y[right_indices]
            # prevent edge case for empty tree
            if left_data_x.shape[0] ==0 or right_data_x.shape[0] == 0:
                tree= np.atleast_2d(np.array(["leaf", np.mean(data_y), np.nan, np.nan], dtype=object))
            else:
                left_tree = self.add_evidence(left_data_x,left_data_y)
                right_tree = self.add_evidence(right_data_x,right_data_y)
                root = np.array([index, split_value, 1, left_tree.shape[0] + 1],dtype=object)
                tree = np.vstack((root, left_tree, right_tree))
        self.tree = tree
        return tree
    def query(self, points):
        arr = np.zeros(points.shape[0])
        for i in range(points.shape[0]):
            node=0
            while self.tree[node][0]!="leaf":
                if points[i][int(self.tree[node][0])]<=self.tree[node][1]:
                    node+=1
                else:
                    node+=self.tree[node][3]
                node=int(node)
            arr[i]=self.tree[node][1]
        return arr
