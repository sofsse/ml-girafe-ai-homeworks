import numpy as np

class SimplifiedBaggingRegressor:
    def __init__(self, num_bags, oob=False):
        self.num_bags = num_bags
        self.oob = oob
        
    def _generate_splits(self, data: np.ndarray):
        '''
        Generate indices for every bag and store in self.indices_list list
        '''
        self.indices_list = []
        data_length = len(data)
        for bag in range(self.num_bags):
            indices_bag_i = np.random.choice(np.arange(data_length), size=(data_length,), replace=True)
            self.indices_list.append(indices_bag_i)
            # Your Code Here
        
    def fit(self, model_constructor, data, target):
        '''
        Fit model on every bag.
        Model constructor with no parameters (and with no ()) is passed to this function.
        
        example:
        
        bagging_regressor = SimplifiedBaggingRegressor(num_bags=10, oob=True)
        bagging_regressor.fit(LinearRegression, X, y)
        '''
        self.data = None
        self.target = None
        self._generate_splits(data)
        assert len(set(list(map(len, self.indices_list)))) == 1, 'All bags should be of the same length!'
        assert list(map(len, self.indices_list))[0] == len(data), 'All bags should contain `len(data)` number of elements!'
        self.models_list = []
        for bag in range(self.num_bags):
            model = model_constructor()
            indices_bag_i = self.indices_list[bag]
            data_bag, target_bag = data[indices_bag_i], target[indices_bag_i] # Your Code Here
            self.models_list.append(model.fit(data_bag, target_bag)) # store fitted models here
        if self.oob:
            self.data = data
            self.target = target
        
    def predict(self, data):
        '''
        Get average prediction for every object from passed dataset
        '''
        # Your code here
        y_preds = []
        for model in self.models_list:
            y_p = model.predict(data)
        y_preds.append(y_p)
        y_preds = np.array(y_preds) # (n_models, n_data)

        y_pred = np.mean(y_preds, axis=0)
        return y_pred
    
    def _get_oob_predictions_from_every_model(self):
        '''
        Generates list of lists, where list i contains predictions for self.data[i] object
        from all models, which have not seen this object during training phase
        '''
        list_of_predictions_lists = [[] for _ in range(len(self.data))]
        # Your Code Here
        for i, data_i in enumerate(self.data):
            for idxs, model in zip(self.indices_list, self.models_list):
                if i not in idxs:
                    list_of_predictions_lists[i].append(float(model.predict(data_i.reshape(1, -1))))

        self.list_of_predictions_lists = np.array(list_of_predictions_lists, dtype=object)
    
    def _get_averaged_oob_predictions(self):
        '''
        Compute average prediction for every object from training set.
        If object has been used in all bags on training phase, return None instead of prediction
        '''
        self._get_oob_predictions_from_every_model()
        self.oob_predictions = np.array([np.mean(di) if len(di)!=0 else np.nan for di in self.list_of_predictions_lists]) # Your Code Here
        
        
    def OOB_score(self):
        '''
        Compute mean square error for all objects, which have at least one prediction
        '''
        self._get_averaged_oob_predictions()
        return np.nanmean((self.target - self.oob_predictions)**2) # Your Code Here