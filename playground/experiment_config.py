# Evaluation functions
from sklearn.metrics import roc_auc_score, make_scorer
from lift.perc_lift_score import perc_lift_score

# Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Samplers
from imblearn.over_sampling import SMOTE, ADASYN
from wgan.imblearn import GANbalancer

def experiment_config(X, idx_cont=None, idx_cat=None):
    ### Samplers
    scorers = {'auc':make_scorer(roc_auc_score, needs_proba=True),
              'TDLift':make_scorer(perc_lift_score, needs_proba=True, percentile=0.1)}

    ### Models
    models = []
    models.append(('LR', LogisticRegression(solver='liblinear'), {
        "C": [1, 1e1, 1e2, 1e3, 1e4]
    }))
    models.append(('RF', RandomForestClassifier(), {
        "n_estimators":[500],
        "max_features":[3,6,9,"sqrt"],
        "min_samples_leaf":[20]
    }))

    ### Samplers
    from imblearn.over_sampling import SMOTE, ADASYN
    from wgan.imblearn import GANbalancer

    # SMOTE
    samplers = []
    samplers.append(('unbalanced', None, {}))
    samplers.append(('SMOTE', SMOTE(), {
        'k_neighbors':[5,10,20]
    }))

    ## ADASYN
    # samplers.append(('ADASYN', ADASYN(), {
    #     'n_neighbors':[5,10,20]
    # }))

    # GAN
    if idx_cont is None:
        idx_cont = list(range(X.shape[1]))

    categorical = None
    if idx_cat is not None:
        categorical = [(i,
                        np.max(X[:,i])+1,
                        int(min(15., np.ceil(np.max((X[:,i])+1)/2)))
                       )
                        for i in idx_cat]

    samplers.append(('cGAN', GANbalancer(
            idx_cont=idx_cont, categorical=categorical,
            generator_input=X.shape[1]
    ), {
        'generator_layers' : [[20],[40],[60],[40,40]],
        'critic_layers'    : [[20],[40],[60],[40,40]],
        'n_iter'           : [1000,5000,10000]
    }))

    return scorers, models, samplers
