import numpy as np
import pandas as pd
import json
import itertools

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedShuffleSplit

from wgan.data_loader import TabularDataset

def prepare_data(X, y=None, X_test=None, y_test=None, idx_cont=None, idx_cat=None, cat_levels=None):
    """
    Helper function to split data (X,y) into stratified train/test sets
    and prepare a data loader

    Input
    -----
    X, y: Array
      Numpy arrays with the explanatory and target data
    idx_cont, idx_cat: 1-D array
      Numpy arrays containing the column indices of continuous and categorical
      variables in X, respectively

    Output
    ------
    X_train, X_test, y_train, y_test, Xy_wgan

    Xy_wgan: Pytorch Dataset
      Pytorch dataset with categorical variables one-hot encoded
    """

    if X_test is None:
        splitter = StratifiedShuffleSplit(n_splits=2, test_size=0.25, random_state=123)
        idx_train, idx_test = next(splitter.split(X,y))
        #X_train, X_test, y_train, y_test = train_test_split(X, y,
        #                                                    stratify=y, test_size=0.5, random_state=123)
        X_train, X_test, y_train, y_test = X[idx_train,:], X[idx_test,:], y[idx_train], y[idx_test]
    else:
        X_train = X
        y_train = y

    if idx_cont is not None:
        scaler = MinMaxScaler()
        X_train[:,idx_cont] = scaler.fit_transform(X_train[:,idx_cont])
        X_test[:,idx_cont] = scaler.transform(X_test[:,idx_cont])

    #X_train_majority = X_train[y_train==0,:]
    #X_train_minority = X_train[y_train==1,:]

    #y_wgan = np.zeros([len(y_train),2])
    #y_wgan[y_train==0,0] = 1
    #y_wgan[y_train==1,1] = 1

    if cat_levels is None:
        cat_levels = [len(np.unique(X_train[:,i])) for i in idx_cat]

    # Only minority class data
    Xy_wgan = TabularDataset(X = X_train[:,idx_cont],
                                X_cat = X_train[:,idx_cat],
                                y= y,
                                cat_levels = cat_levels
                           )

    # Make sure the order of the training data is the same as for the data loader
    idx_order = list(itertools.chain.from_iterable([x for x in [idx_cont,idx_cat]\
                                                      if x is not None ]))

    # Update the cont and cat indices to the new order
    # Updating order is helpful to compare the synthetic and original data directly
    # TODO: Implement reordering to original order for synthetic data
    if idx_cont is not None:
        idx_cont = list(range(0,len(idx_cont)))
    if idx_cat is not None:
        temp = len(idx_cont) if idx_cont is not None else 0
        idx_cat = list(range(temp,temp+len(idx_cat)))

    return X_train[:,idx_order], X_test[:,idx_order], y_train, y_test, Xy_wgan, idx_cont, idx_cat, scaler


def load_DMC02(path):
    """
    Load dataset of the Data Mining Cup 2012

    Output
    ------
    X, y, idx_cont, idx_cat
    X: Explanatory variabes
    y: Target variable (0/1)
    idx_cont: Column indices of continuous variables
    idx_cat: Column indices of categorical variables
    """
    with open(path) as file:
        data_list = json.load(file)
    # Take y values
    y = np.array(data_list['Y']).T.flatten()
    # Rearrange from -1,1 to 0,1
    y[y==-1] = 0
    # Get X values
    X = pd.DataFrame(data_list["Xcat"]).T.values
    # Hard-code continuous and categorical variable indices
    idx_cat = np.array([0,1,2])
    idx_cont = np.arange(3,X.shape[1])
    return X, y, idx_cont, idx_cat

def load_DMC10(path):
    """
    Load dataset of the Data Mining Cup 2010

    Output
    ------
    X_train, X_test, y_train, y_test, idx_cont, idx_cat
    X_[train/test]: Explanatory variabes
    y_[train/test]: Target variable (0/1)
    idx_cont: Column indices of continuous variables
    idx_cat: Column indices of categorical variables
    """
    data_train = pd.read_csv(path+"/dmc2010_train.txt", sep=";", low_memory=False,
                   dtype={"invoicepostcode":'str', 'delivpostcode':'str', 'advertisingdatacode':'str'})

    X_test = pd.read_csv(path+"/dmc2010_class.txt", sep=";", low_memory=False,
                   dtype={"invoicepostcode":'str', 'delivpostcode':'str', 'advertisingdatacode':'str'})
    y_test = pd.read_csv(path+"/dmc2010_real.txt", sep=";", header=None, names=['customernumber','target90'] )
    data_test = pd.merge(X_test,y_test, on="customernumber")

    for data in [data_train, data_test]:

        data.loc[data.delivpostcode.isna(), "delivpostcode"] = "MISSING"
        data.loc[data.advertisingdatacode.isna(), "advertisingdatacode"] = "NO_CODE"

        data.loc[data.delivpostcode.isna(), "delivpostcode"] = "MISSING"
        data.loc[data.advertisingdatacode.isna(), "advertisingdatacode"] = "NO_CODE"

        ## Deal with dates
        data["date"] = pd.to_datetime(data.date)
        data["datecreated"] = pd.to_datetime(data.datecreated)

        data["deliverydatepromised"] = pd.to_datetime(data.deliverydatepromised, errors='coerce')
        data["deliverydatereal"] = pd.to_datetime(data.deliverydatereal, errors='coerce')
        # For year=0000 or year=4576, replace weird date with other delivery date
        # or if impossible both with date of first order
        data["deliverydatepromised"].fillna(data.deliverydatereal, inplace=True)
        data["deliverydatereal"].fillna(data.deliverydatepromised, inplace=True)
        data["deliverydatepromised"].fillna(data.date, inplace=True)
        data["deliverydatereal"].fillna(data.date, inplace=True)

        #data["first_order_year"] = data.date.dt.year
        data["first_order_month"] = data.date.dt.month.astype(int)
        #data["first_order_day"] = data.date.dt.day.astype(int)

        #data["deliverydatepromised_month"] = data.deliverydatepromised.dt.month
        #data["deliverydatepromised_day"] = data.deliverydatepromised.dt.day
        #data["deliverydatepromised_weekday"] = data.deliverydatepromised.dt.dayofweek

        data["deliverydatereal_month"] = data.deliverydatereal.dt.month.astype(int)
        data["deliverydatereal_day"] = data.deliverydatereal.dt.day.astype(int)
        data["deliverydatereal_weekday"] = data.deliverydatepromised.dt.dayofweek.astype(int)

        data["account_age"] = (data.deliverydatereal - data.datecreated).dt.days
        data["firstorderdelay"] = (data.date - data.datecreated).dt.days
        data["deliverydelay"] = (data.deliverydatereal - data.deliverydatepromised).dt.days

        # Drop unique identifier and processed raw date variables
        data.drop(["customernumber","date","datecreated", "deliverydatepromised",\
                   "deliverydatereal",
                   "points" # seems to be all zeros
                   ], axis=1, inplace=True)

    ## Deal with categorical
    # Unify rare values in advertisingdatacode
    for cat_var in ["invoicepostcode","delivpostcode","advertisingdatacode"]:
        counts = data_train[cat_var].value_counts()
        rare = list(counts.index[counts<60])
        test_only = [x for x in list(data_test[cat_var].unique())
                       if x not in list(data_train[cat_var].unique())]
        level_dict = {level:"___" for level in rare+test_only}
        data_train[cat_var] = data_train[cat_var].replace(level_dict)
        data_test[cat_var]  =  data_test[cat_var].replace(level_dict)

    # Map to integer
    cat_dict = {}
    for cat_var in ["model","delivpostcode", "invoicepostcode","advertisingdatacode",
                     "first_order_month", #"first_order_day",
                     "deliverydatereal_month","deliverydatereal_day",
                     "deliverydatereal_weekday"]:
        cat_dict[cat_var] = {level:level_int for level_int,level in enumerate(data_train[cat_var].unique())}
        data_train[cat_var] = data_train[cat_var].map(cat_dict[cat_var])
        data_test[cat_var] = data_test[cat_var].map(cat_dict[cat_var])
    #
    #     data[cat_var] = data[cat_var].astype("category").cat.codes

    # Separate target variable
    y_train = data_train.pop('target90')
    y_test = data_test.pop('target90')

    # Cont/Cat variables
    cat_names = ['salutation', 'title', 'domain', 'newsletter', 'model', 'paymenttype',
       'deliverytype', 'invoicepostcode', 'delivpostcode', 'voucher',
       'advertisingdatacode', 'gift', 'entry', 'shippingcosts', 'first_order_month',
       'deliverydatereal_month', 'deliverydatereal_day',
       'deliverydatereal_weekday']
    idx_cont = [i for i, var in enumerate(data_train.columns) if var not in cat_names]
    idx_cat =  [i for i, var in enumerate(data_train.columns) if var in cat_names]

    return data_train.to_numpy(), data_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy(), idx_cont, idx_cat, cat_dict

def load_KDD09(path):
    pd.read_csv(path, sep="\t")
    return None
