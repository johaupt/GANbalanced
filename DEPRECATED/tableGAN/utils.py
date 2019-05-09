import numpy as np
import scipy.stats as sp
import pandas as pd
import torch

def create_continuous_data(N, pos_ratio=0, k=10, random_state=None):
    if random_state is not None: np.random.seed(random_state)
    # Group indicator
    #group = sp.binom.rvs(p=0.25, n=1, size=N)
    N_neg = int(N*(1-pos_ratio))
    N_pos = N-N_neg
    y = np.concatenate([np.zeros(N_neg), np.ones(N_pos)])

    mean = np.random.uniform(size=k)
    mean0 = np.random.normal(loc=mean,scale=0.1)
    mean1 = np.random.normal(loc=mean,scale=0.1)
    cov = sp.invwishart.rvs(df=k, scale=np.eye(k)/2)
    # Continuous variables
    X0 = sp.multivariate_normal.rvs(mean=mean0, cov= cov, size=N_neg)
    X1 = sp.multivariate_normal.rvs(mean=mean1, cov= cov, size=N_pos)

    X = np.vstack([X0,X1])

    return {"X":X, "y":y,"mean0":mean0,"mean1":mean1, "cov":cov}


def create_GAN_data(N, class_ratio=0.5, random_state=None):
    """
    Create example dataset with variables randomly drawn from different distributions
    to mimic a real dataset with continuous, binary and categorical data

    Output
    ------
    Pandas dataframe
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


from torch.utils.data import Dataset, DataLoader

class TabularDataset(Dataset):
  def __init__(self, data, cat_cols=None, output_col=None):
    """
    Characterizes a Dataset for PyTorch

    Parameters
    ----------

    data: pandas data frame
      The data frame object for the input data. It must
      contain all the continuous, categorical and the
      output columns to be used.

    cat_cols: List of strings
      The names of the categorical columns in the data.
      These columns will be passed through the embedding
      layers in the model. These columns must be
      label encoded beforehand.

    output_col: string
      The name of the output variable column in the data
      provided.
    """

    self.n = data.shape[0]

    if output_col: # if and string defaults to if not None
      self.y = data[output_col].astype(np.float32).values.reshape(-1, 1)
    else:
      self.y =  np.zeros((self.n, 1))

    self.cat_cols = cat_cols if cat_cols else []
    self.cont_cols = [col for col in data.columns
                      if col not in self.cat_cols + [output_col]]

    if self.cont_cols:
      self.cont_X = data[self.cont_cols].astype(np.float32).values
    else:
      self.cont_X = np.zeros((self.n, 1))

    if self.cat_cols:
      self.cat_X = data[cat_cols].astype(np.int64).values
    else:
      self.cat_X =  np.zeros((self.n, 1))

  def __len__(self):
    """
    Denotes the total number of samples.
    """
    return self.n

  def __getitem__(self, idx):
    """
    Generates one sample of data.
    """
    return [self.y[idx], self.cont_X[idx], self.cat_X[idx]]


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder

class Preprocessor():
    """
    Class to scale and one-hot encode data and undo preprocessing after data generation
    """
    def __init__(self, numeric_features=None, binary_features=None, categorical_features=None):
        self.scaler = MinMaxScaler()
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
