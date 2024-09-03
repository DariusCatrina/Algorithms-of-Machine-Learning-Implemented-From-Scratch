'''
Backbones for KNN:
    -brute force
    -KD tree
    -Ball tree(soon)

'''


import heapq
from dataclasses import dataclass

'''
Brute force class that stores the points in a priority queue with the priority value set as
the distance between the respective point and the query point

++ struct that hold the features of a datapoint + label
'''


@dataclass
class DataPoint:
    features: list[int]
    label: bool
        

class BruteForce(object):
    def __init__(self, data_points, y):
        self.data_points : list[DataPoint] = [] 
        for feature,label  in zip(data_points, y):
            self.data_points.append(DataPoint(features=feature, label=label))

    def query(self, query_point, n_neighbors, dist_func):
        query_point = DataPoint(features=query_point, label=None)
        self.best_node_queue = []
        
        for p in self.data_points:            
            heapq.heappush(self.best_node_queue, (dist_func(p.features, query_point.features), p))

        neighbors = []
        for i in range(n_neighbors):
            n = int(heapq.heappop(self.best_node_queue)[-1].label)
            neighbors.append(n)

        return neighbors
    
'''
Node class for the KD Tree data structure
'''
class Node(object):
    def __init__(self, root=None, left_child=None, right_child=None, label=None) -> None:
        self.root = root
        self.left_child = left_child
        self.right_child = right_child
        self.label = label

    @staticmethod
    def print_tree(tree, level=0):
        if tree != None:
            Node.print_tree(tree.left_child, level + 1)
            print(' ' * 12 * level + '-> ' + str(tree.root))
            Node.print_tree(tree.right_child, level + 1)

'''
Implementation from scratch of the KD tree dara structure
'''
class KDTree(object):
    def __init__(self, data_points, y, k=None):
        self.data_points = data_points
        self.y = y

        if k == None:
            self.k = len(self.data_points[0])
        else:
            self.k = k
    
        
    def build_tree(self):
        self.point_class_dict = {}

        for p, l in zip(self.data_points, self.y):
            self.point_class_dict[tuple(p)] = l

        self.kd_tree = self._build_tree(curr_points=self.data_points, depth=0)


    def _build_tree(self, curr_points, depth):
        
        
        if len(curr_points) == 0:
            return
        if len(curr_points) == 1:
            return Node(root=curr_points[0], label=self.point_class_dict[tuple(curr_points[0])])
        
        axis = depth % self.k

        sorted_points = sorted(curr_points, key=lambda x:x[axis])
        median_index = int(len(sorted_points) / 2)

        median = sorted_points[median_index]

        root_val = median
        tree = Node(root=root_val, label=self.point_class_dict[tuple(root_val)])

        tree.left_child = self._build_tree(sorted_points[:median_index], depth+1)
        tree.right_child = self._build_tree(sorted_points[median_index + 1:], depth+1)

        return tree
    
    def query(self, query_point, n_neighbors, dist_func):
        self.query_point = Node(root=query_point)
        self.dist = dist_func
        self.best_node_queue = []
        self.best_node = self.kd_tree
        self.best_dist = self.dist(self.best_node.root, self.query_point.root)

        self._query(self.kd_tree, 0)
        neighbors = []

        for _ in range(n_neighbors):
            neighbors.append(heapq.heappop(self.best_node_queue)[-1].label)

        return neighbors

    def _query(self, curr_tree, depth):
       
        axis = depth % self.k
        
        if curr_tree == None:
            return 
        else:
            curr_dist = self.dist(curr_tree.root, self.query_point.root)
            heapq.heappush(self.best_node_queue, (curr_dist, curr_tree))
        
        
        if curr_dist< self.best_dist:
            self.best_node = curr_tree
            self.best_dist = curr_dist

        diff = self.query_point.root[axis] - curr_tree.root[axis]
        if diff <= 0:
            good_side = curr_tree.left_child
            bad_side = curr_tree.right_child
        else:
            good_side = curr_tree.right_child
            bad_side = curr_tree.left_child

        self._query(good_side, depth+1)
        if diff**2 < self.best_dist:
            self._query(bad_side, depth+1)
    
        return 