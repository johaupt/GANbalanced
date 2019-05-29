import numpy as np
from copy import deepcopy
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

from imblearn import FunctionSampler

import matplotlib.pyplot as plt
import seaborn as sns

################################################################################

def upsampling_performance(X_train, X_test, y_train, y_test, classifier, sampler, scorer):
    """
    Train classifier on resampled data and evaluate performance in terms of AUC

    Input
    -----
    sampler: imblearn object or None
    """
    model = deepcopy(classifier)

    # Sample synthetic SMOTE data
    if sampler:
        X_train, y_train =  sampler.fit_resample(X_train,y_train)

    model.fit(X=X_train, y=y_train)
    prob = model.predict_proba(X_test)[:,1]
    imb_ratio = np.mean(y_train)

    return [scorer_func(y_true=y_test, y_score=prob) for scorer_func in scorer] 

################################################################################

def discriminator_evaluation(X_true, X_fake, classifier):
    """
    Train classifier to differentiate between X_true and X_fake. Evaluate
    performance of classifier (on same data) using AUC ROC and accuracy

    Input
    -----
    X_true, X_false: array

    classifier: sklearn classifier
      Classifier to try and separate true and fake data. The model object is
      copied before fitting to avoid side-effects.

    Output
    ------
    AUC ROC (float), accuracy (float)
    """
    model_fakereal = deepcopy(classifier)
    # Merge fake and real data
    X_fakereal = np.vstack([X_true,
                            X_fake])
    # Create target variable (0:real/1:fake)
    y_fakereal = np.concatenate([np.zeros(X_true.shape[0]),
                                  np.ones(X_fake.shape[0])]
                                  ).flatten()

    model_fakereal.fit(X_fakereal, y_fakereal)

    pred_fakereal = model_fakereal.predict_proba(X_fakereal)[:,1]
    return roc_auc_score(y_fakereal, pred_fakereal),\
           accuracy_score(y_fakereal, pred_fakereal>0.5)


################################################################################

def plot_distributions(X_true, X_fake, y_true=None, y_fake=None):
    """
    Plot a k x k matrix to compare distributions of the variables in X
    """
    no_vars = X_true.shape[1]
    combinations = [(x,y) for x in range(no_vars) for y in range(no_vars) if y>x]

    if y_true:
        minority_true = X_true[y_true==1,:]
        minority_fake = X_fake[y_fake==1,:]
        majority_true = X_true[y_true==0,:]
        majority_fake = X_fake[y_fake==0,:]
    else:
        minority_true = X_true
        minority_fake = X_fake


    fig, axes = plt.subplots(nrows=no_vars, ncols=no_vars, sharex=True,\
     squeeze=True,figsize=(10,10))
    for y in axes:
        for x in y:
            x.set_xticklabels([])
            x.set_yticklabels([])

    # Plot univariate minority distribution on diagonal
    for i in range(no_vars):
        print(f"Plotting univariate distribution {i+1}/{no_vars}")
        sns.kdeplot(minority_true[:,i], alpha=0.5, shade=True, color="blue",\
         ax=axes[(i,i)])
        sns.kdeplot(minority_fake[:,i], alpha=0.5, shade=True, color="green",\
         ax=axes[(i,i)])

    # Plot conditional distributions in the lower and upper triangles
    for i,j in combinations:
        print(f"Plotting univariate distribution {i},{j}")
        axes[(i,j)].set_ylim(0,1)
        # majority (upper right)
        if y_true is not None:
            sns.kdeplot(majority_real[0:1000,i], majority_real[0:1000,j],\
               alpha=0.5, cmap="Blues", ax=axes[(i,j)])
            sns.kdeplot(majority_fake[:,i], majority_fake[:,j],\
               alpha=0.5, cmap="Greens", ax=axes[(i,j)], )

        # minority (lower left)
        sns.kdeplot(minority_true[:,i], minority_true[:,j], alpha=0.5,\
            cmap="Blues", ax=axes[(j,i)])
        sns.kdeplot(minority_fake[:,i], minority_fake[:,j], alpha=0.5,\
            cmap="Greens", ax=axes[(j,i)])

    return fig
