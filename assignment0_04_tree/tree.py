import numpy as np
from sklearn.base import BaseEstimator

def entropy(y):  
    """
    Computes entropy of the provided distribution. Use log(value + eps) for numerical stability
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, n_classes)
        One-hot representation of class labels for corresponding subset
    
    Returns
    -------
    float
        Entropy of the provided subset
    """
    EPS = 0.0005

    # YOUR CODE HERE
    p = np.sum(y, axis=0)/np.sum(y)
    return -np.sum(p*np.log(p+EPS))
    
def gini(y):
    """
    Computes the Gini impurity of the provided distribution
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, n_classes)
        One-hot representation of class labels for corresponding subset
    
    Returns
    -------
    float
        Gini impurity of the provided subset
    """

    # YOUR CODE HERE
    p = np.sum(y, axis=0)/np.sum(y)
    return 1 - np.sum(p**2)
    
def variance(y):
    """
    Computes the variance the provided target values subset
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, 1)
        Target values vector
    
    Returns
    -------
    float
        Variance of the provided target vector
    """
    
    # YOUR CODE HERE
    return np.var(y)

def mad_median(y):
    """
    Computes the mean absolute deviation from the median in the
    provided target values subset
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, 1)
        Target values vector
    
    Returns
    -------
    float
        Mean absolute deviation from the median in the provided vector
    """

    # YOUR CODE HERE
    
    return np.abs(y-np.median(y)).mean()


def one_hot_encode(n_classes, y):
    y_one_hot = np.zeros((len(y), n_classes), dtype=float)
    y_one_hot[np.arange(len(y)), y.astype(int)[:, 0]] = 1.
    return y_one_hot


def one_hot_decode(y_one_hot):
    return y_one_hot.argmax(axis=1)[:, None]


class Node:
    """
    This class is provided "as is" and it is not mandatory to use it in your code.
    """
    def __init__(self, feature_index, threshold, proba=0, left_child=None, right_child=None):
        self.feature_index = feature_index
        self.value = threshold
        self.proba = proba
        self.left_child = left_child
        self.right_child = right_child
        
        
class DecisionTree(BaseEstimator):
    all_criterions = {
        'gini': (gini, True), # (criterion, classification flag)
        'entropy': (entropy, True),
        'variance': (variance, False),
        'mad_median': (mad_median, False)
    }

    def __init__(self, n_classes=None, max_depth=np.inf, min_samples_split=2, 
                 criterion_name='gini', debug=False):

        assert criterion_name in self.all_criterions.keys(), 'Criterion name must be on of the following: {}'.format(self.all_criterions.keys())
        
        self.n_classes = n_classes
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion_name = criterion_name
        self.criterion, self.classification = self.all_criterions[self.criterion_name]

        self.depth = 0
        self.root = None # Use the Node class to initialize it later
        self.debug = debug


        
    def make_split(self, feature_index, threshold, X_subset, y_subset):
        """
        Makes split of the provided data subset and target values using provided feature and threshold
        
        Parameters
        ----------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels for corresponding subset
        
        Returns
        -------
        (X_left, y_left) : tuple of np.arrays of same type as input X_subset and y_subset
            Part of the providev subset where selected feature x^j < threshold
        (X_right, y_right) : tuple of np.arrays of same type as input X_subset and y_subset
            Part of the providev subset where selected feature x^j >= threshold
        """

        # YOUR CODE HERE
        X_left, y_left = X_subset[X_subset[:, feature_index]<threshold, :], y_subset[X_subset[:, feature_index]<threshold, :]
        X_right, y_right = X_subset[X_subset[:, feature_index]>=threshold, :], y_subset[X_subset[:, feature_index]>=threshold, :]
        
        return (X_left, y_left), (X_right, y_right)
    
    def make_split_only_y(self, feature_index, threshold, X_subset, y_subset):
        """
        Split only target values into two subsets with specified feature and threshold
        
        Parameters
        ----------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels for corresponding subset
        
        Returns
        -------
        y_left : np.array of type float with shape (n_objects_left, n_classes) in classification 
                   (n_objects, 1) in regression 
            Part of the provided subset where selected feature x^j < threshold

        y_right : np.array of type float with shape (n_objects_right, n_classes) in classification 
                   (n_objects, 1) in regression 
            Part of the provided subset where selected feature x^j >= threshold
        """

        # YOUR CODE HERE
        y_left = y_subset[X_subset[:, feature_index]<threshold, :] 
        y_right = y_subset[X_subset[:, feature_index]>=threshold, :]
        
        return y_left, y_right

    def choose_best_split(self, X_subset, y_subset):
        """
        Greedily select the best feature and best threshold w.r.t. selected criterion
        
        Parameters
        ----------
        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels or target values for corresponding subset
        
        Returns
        -------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        """
        # YOUR CODE HERE

        feature_index, threshold = None, None
        (X_ll, y_ll), (X_rr, y_rr) = (None, None), (None, None)
        dG = -np.inf
        HQ = self.criterion(y_subset)

        for fi in range(X_subset.shape[-1]):
            thresholds = np.unique(X_subset[:, fi])[1:]
            #thresholds = np.vstack([thresholds[1:], thresholds[:-1]]).mean(axis=0) # average between boundaries like how it implemented in sklearn
            for th in thresholds:
                (X_l, y_l), (X_r, y_r) = self.make_split(fi, th, X_subset, y_subset)
                HL = self.criterion(y_l)
                HR = self.criterion(y_r)
                dg_trial = HQ - len(X_l)/len(X_subset) * HL - len(X_r)/len(X_subset) * HR # wanna make it as small as possible

                if dg_trial > dG: 
                    dG = dg_trial
                    feature_index = fi
                    threshold = th
                    (X_ll, y_ll), (X_rr, y_rr) = (X_l, y_l), (X_r, y_r)

        return feature_index, threshold, (X_ll, y_ll), (X_rr, y_rr)
    
    def make_tree(self, X_subset, y_subset, depth=0):
        """
        Recursively builds the tree
        
        Parameters
        ----------
        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels or target values for corresponding subset
        
        Returns
        -------
        root_node : Node class instance
            Node of the root of the fitted tree
        """
        # YOUR CODE HERE
        if self.classification:
            y_proba = np.sum(y_subset, axis=0)/np.sum(y_subset) # probability
        else:
            y_proba = np.median(y_subset) #.mean()

        if depth == self.max_depth or len(np.unique(y_subset, axis=0)) == 1 or len(X_subset) < self.min_samples_split: # reached max depth or pure node or node is small enough
            return Node(feature_index=None, threshold=None, proba=y_proba, left_child=None, right_child=None) # end of recursion, return 

        # otherwise make a split and 
        feature_index, threshold, (X_l, y_l), (X_r, y_r) = self.choose_best_split(X_subset, y_subset)
        if (feature_index is not None) and (threshold is not None):
            lc = self.make_tree(X_l, y_l, depth=1+depth)
            rc = self.make_tree(X_r, y_r, depth=1+depth)
            new_node = Node(feature_index, threshold, proba=y_proba, left_child=lc, right_child=rc)
            #self.depth = self.get_depth(new_node)
            return new_node
            
    def get_depth(self, node):
        if node: 
            return 1 + max(self.get_depth(node.left_child), self.get_depth(node.right_child))
        else:
            return 0


    def fit(self, X, y):
        """
        Fit the model from scratch using the provided data
        
        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data to train on

        y : np.array of type int with shape (n_objects, 1) in classification 
                   of type float with shape (n_objects, 1) in regression 
            Column vector of class labels in classification or target values in regression
        
        """
        assert len(y.shape) == 2 and len(y) == len(X), 'Wrong y shape'
        self.criterion, self.classification = self.all_criterions[self.criterion_name]
        if self.classification:
            if self.n_classes is None:
                self.n_classes = len(np.unique(y))
            y = one_hot_encode(self.n_classes, y)

        self.root = self.make_tree(X, y)
    
    def predict(self, X):
        """
        Predict the target value or class label  the model from scratch using the provided data
        
        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data the predictions should be provided for

        Returns
        -------
        y_predicted : np.array of type int with shape (n_objects, 1) in classification 
                   (n_objects, 1) in regression 
            Column vector of class labels in classification or target values in regression
        
        """
        # YOUR CODE HERE
        y_predicted = []
        for x in X:
            tree = self.root
            while (tree.left_child is not None) or (tree.right_child is not None):
                if tree.value:
                    if x[tree.feature_index]<tree.value:
                        tree = tree.left_child
                    else:
                        tree = tree.right_child
                else:
                    break 

            if self.classification:
                y_predicted.append(np.argmax(tree.proba))
            else:
                y_predicted.append(tree.proba)

        return y_predicted
        
    def predict_proba(self, X):
        """
        Only for classification
        Predict the class probabilities using the provided data
        
        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data the predictions should be provided for

        Returns
        -------
        y_predicted_probs : np.array of type float with shape (n_objects, n_classes)
            Probabilities of each class for the provided objects
        
        """
        assert self.classification, 'Available only for classification problem'

        # YOUR CODE HERE
        y_predicted_probs = []
        for x in X:
            tree = self.root
            while (tree.left_child is not None) or (tree.right_child is not None):
                if tree.value:
                    if x[tree.feature_index]<tree.value:
                        tree = tree.left_child
                    else:
                        tree = tree.right_child
                else:
                    break 

            y_predicted_probs.append(tree.proba)
        
        return np.array(y_predicted_probs).reshape(len(X), self.n_classes)
