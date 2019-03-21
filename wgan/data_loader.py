import numpy as np
from torch.utils.data import DataLoader, Dataset

class TabularDataset(Dataset):
    def __init__(self, X, X_cat=None, y=None, cat_levels=None, group_filter=None):
        """
        Characterizes a Dataset for PyTorch WGAN including categorical
        and/or auxiliary variables

        Parameters
        ----------

        X: numpy array
          The array object for the input data.

        X_cat: numpy array
          Array containing categorical data in one column, each level coded as
          an integer [0;no. of levels]. Categorical variables will be transformed
          to one-hot encoding automatically.

        y: 2-D array-like
          array of length equal to the number of rows in X, where each column is
          a categorical auxiliary variable coded as integer [0;no. of levels].
          Will be transformed to one-hot encoding automatically.

        cat_levels: list of integers
          The number of levels in each categorical variable X_cat. If None then
          the number of levels is guessed from the number of unique values in
          each categorical column.
        """
        self.no_cat = 0
        self.no_cont = X.shape[1]

        if y is not None:
            try:
                y_cols = y.shape[1]
            except:
             y = y.reshape(-1,1)
             y_cols = 1
            y_levels = [len(np.unique(y[:,i])) for i in range(y_cols)]

            # One hot encode y's
            self.y = np.hstack([np.eye(y_levels[i])[y[:,i].astype(np.int32)].astype(np.float32)
                                for i in range(y.shape[1])])
            self.no_aux = self.y.shape[1]
        else:
            self.y = np.zeros((self.n, 1)).astype(np.float32)
            self.no_aux = 0

        if X_cat is not None:
            if cat_levels == 0:
                cat_levels = [len(np.nunique(X_cat[:,i])) for i in range(X_cat.shape[1])]
            X_cat = [np.eye(cat_levels[i])[X_cat[:,i].astype(np.int32)].astype(np.float32) for i in range(X_cat.shape[1])]
            X = np.hstack([X, *X_cat])
            self.no_cat = len(cat_levels)

        self.X = X.astype(np.float32)
        self.cat_levels = cat_levels

        # Filter training data for specific groups
        if group_filter is not None:
            if y is None:
                raise ValueError("'y' must be given if group_filter specified")
            if y_cols >1:
                raise NotImplementedError("Filtering for several y's not implemented")
            self.X = self.X[np.isin(y, group_filter).flatten(),:]
            self.y = self.y[np.isin(y, group_filter).flatten(),:]

        self.n = self.X.shape[0]


    def __len__(self):
        """
        Denotes the total number of samples.
        """
        return self.n

    def __getitem__(self, idx):
        """
        Generates one sample of data.
        """
        return [self.X[idx,:], self.y[idx,:]]


class CategoricalDataset(Dataset):
    def __init__(self, X, X_cat=None, y=None):
        """
        Characterizes a Dataset for PyTorch

        Parameters
        ----------

        X: numpy array
          The array object for the input data.

        X_cat:

        y: array-like
          The name of the output variable column in the data
          provided or an array of length equal to the number of
          rows in X.
        """
        self.n = X.shape[0]
        self.X = X.astype(np.float32)

        if y is not None:
            self.y = y.astype(np.float32).reshape(-1, 1)
        else:
            self.y = np.zeros((self.n, 1)).astype(np.float32)

        if X_cat is not None:
            self.X_cat = [X_cat[:,i].astype(np.int64) for i in range(X_cat.shape[1])]
            self.X_cat = [np.eye(len(np.unique(var)))[var].astype(np.float32) for var in self.X_cat]
        else:
            self.X_cat =  [np.zeros((self.n, 1))]

    def __len__(self):
        """
        Denotes the total number of samples.
        """
        return self.n

    def __getitem__(self, idx):
        """
        Generates one sample of data.
        """
        return [self.X[idx], [cat[idx] for cat in self.X_cat], self.y[idx],]
