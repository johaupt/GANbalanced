import numpy as np
import scipy.stats as sp
import pandas as pd

def create_GAN_data(N, class_ratio=0.5, random_state=None):
    """
    Create example dataset with variables randomly drawn from different distributions
    to mimic a real dataset with continuous, binary and categorical data
    Output: Pandas dataframe
    """
    if random_state is not None: np.random.seed(random_state)
    # Group indicator
    #group = sp.binom.rvs(p=0.25, n=1, size=N)
    group = np.concatenate([np.zeros([int(N*(1-class_ratio))]), np.ones([int(N*class_ratio)])])

    # Continuous variables
    x0 = sp.poisson.rvs(mu=np.where(group==1,1.,2.))
    x1 = sp.norm.rvs(loc=np.where(group==1,-2,2),
                     scale=1)
    x2 = sp.norm.rvs(loc=x1,
                     scale=1)
    x3 = (x0**2 + x1**2)
    x456 = sp.multivariate_normal.rvs(mean=[0,0,0], cov= np.stack(([1,0.8,0.2], [0.8,1.,0.], [0.2,0.,1.]),axis=0), size=N)

    # Discrete variables
    # Binary
    x7 = sp.binom.rvs(p=np.where(group==1,0.6,0.3),n=1)
    # Three class
    x890_0 = sp.multinomial.rvs(p=[0.7,0.2,0.1],n=1,size=N)
    x890_1 = sp.multinomial.rvs(p=[0.2,0.7,0.1],n=1,size=N)
    x890 = np.zeros([N,3])
    for i in range(N):
        if group[i]==1:
            x890[i,] = x890_1[i,]
        else:
            x890[i,] = x890_0[i,]


    data = pd.DataFrame(np.column_stack([x0,x1,x2,x3,x456,group,x7,x890]))
    data.rename({7:"group"}, axis="columns", inplace=True)
    return data


from torch.utils.data import Dataset
class TabularDataset(Dataset):
    """
    Data class for pytorch data loader for tabular data
    data (array-like): numpy array or DataFrame with the data
    """

    def __init__(self, data, label):
        self.data_frame = pd.DataFrame(data)
        self.label = label

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        #observation = self.data_frame.iloc[idx, 1:].values
        observation = self.data_frame.iloc[idx, :].values
        label = []
        if self.label is not None:
            label = self.label[idx]
        return (observation, label)


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class Preprocessor():
    """
    Class to scale and one-hot encode data and undo preprocessing after data generation
    """
    def __init__(self, numeric_features=None, binary_features=None, categorical_features=None):
        self.scaler = StandardScaler()
        self.transformer = OneHotEncoder(handle_unknown='ignore', sparse=False)

        # Remember the order of the variables to put them back to their original index
        self.var_order = np.argsort([*numeric_features, *binary_features, *categorical_features])
        self.numeric_org = numeric_features
        self.binary_org = binary_features
        self.categorical_org = categorical_features

        self.numeric_new = None
        self.binary_new = None
        self.categorical_new = None

    def fit_transform(self, X):
        """
        Fit scaler and one-hot encoder to data and apply transformation
        Input: numpy array
        Output: numpy array, variables in order numeric, binary, categorical
        Notes: Categorical variables that are already one-hot encoded should be
               treated as binary variables and will not be transformed
        """
        X_num, X_cat, X_bin = [np.empty(shape = (X.shape[0],)) for i in range(3)]
        i = 0

        # Fit the transformer to each type of data
        # Remember the new indices of the variable types for the inverse transform
        if self.numeric_org is not None:
            X_num = self.scaler.fit_transform(X[:, self.numeric_org])
            self.numeric_new = list(range(len(self.numeric_org)))
            i += len(self.numeric_org)
        if self.binary_org is not None:
            X_bin = X[:,self.binary_org]
            self.binary_new = list(range(i, i+len(self.binary_org)))
            i += len(self.binary_org)
        if self.categorical_org is not None:
            X_cat = self.transformer.fit_transform(X[:, self.categorical_org])
            self.categorical_new = list(range(i, i+X_cat.shape[1]))

        return np.hstack([X_num, X_bin, X_cat])

    def inverse_transform(self, X):
        """
        Reverse the scaling and one-hot encoding to create data in the original
        input format and in the original order of variables.
        """
        X_num, X_cat, X_bin = [np.empty(shape = (X.shape[0],)) for i in range(3)]

        if self.numeric_new is not None:
            X_num = self.scaler.inverse_transform(X[:, self.numeric_new])
        if self.binary_new is not None:
            X_bin = X[:,self.binary_new]
        if self.categorical_new is not None:
            X_cat = self.transformer.inverse_transform(X[:, self.categorical_new])

        X = np.hstack([X_num, X_bin, X_cat])[:,self.var_order]

        return X
