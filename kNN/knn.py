from _back_bone import KDTree, Node, BruteForce

from math import dist
import numpy as np

'''
Wrapper class to apply the KNN classification pipeline
'''
class KNNClassifier(object):
    def __init__(self,  n_neighbors, algorithm, metric='euclidian') -> None:
        self.dist_func = self._dist_func(metric)
        self.n_neighbors = n_neighbors

        self.algorithm = algorithm

    def fit(self, train_points, classes):
        
        if self.algorithm == 'kd_tree':
            self.data_struct = KDTree(data_points=train_points, y=classes)
            self.data_struct.build_tree()

        elif self.algorithm == 'brute_force':
            self.data_struct = BruteForce(data_points=train_points, y=classes)

    def predict(self, query_points): 
        y_pred = []
        for query_point in query_points:
                classes = self.data_struct.query(query_point, self.n_neighbors, self.dist_func)

                num_pos = sum(classes)
                num_neg = len(classes) - num_pos

                if num_pos/len(classes) > 0.5:
                    y_pred.append(1)
                else:
                    y_pred.append(0)

        return y_pred

    def _dist_func(self, metric):
        if metric == 'euclidean':
            return KNNClassifier._euclidian_dist
        
        elif metric == 'manhattan':
            return KNNClassifier._manhattan_dist
        
        elif metric == 'cosine':
            return KNNClassifier._cosine_sim_dist

        raise NotImplementedError()
        

    @staticmethod
    def _euclidian_dist(a, b):
        return dist(a,b)
    
    @staticmethod
    def _manhattan_dist(a,b):
        return sum(abs(val1-val2) for val1, val2 in zip(a,b))
    
    @staticmethod
    def _cosine_sim_dist(a,b):
        return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))



