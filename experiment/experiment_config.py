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
    ### Models
    models = []
    models.append(('LR', LogisticRegression(solver='liblinear'), {
        "C": [10]
    }))
    # models.append(('RF', RandomForestClassifier(), {
    #     "n_estimators":[200],
    #     "max_features":["sqrt"],
    #     "min_samples_leaf":[20]
    # }))

    ### Samplers
    from imblearn.over_sampling import SMOTE, ADASYN
    from wgan.imblearn import GANbalancer

    samplers = []

    # GAN

    samplers.append(('cGAN', GANbalancer(
            idx_cont=idx_cont, categorical=categorical, batch_size = 128, auxiliary=True
    ), {
        'generator_input'  : [40,100],
        'generator_layers' : [[40,40],[100,100]],
        'critic_layers'    : [[40],[100],[40,40],[100,100]],
        'n_iter'           : [2e5],
        'critic_iterations': [3]
    }))

    #SMOTE
    samplers.append(('SMOTE', SMOTENC(categorical_features=idx_cat), {
        'k_neighbors':[5,10,15,20,25,50,100]
    }))

    # baseline
    samplers.append(('unbalanced', None, {}))



    # # ADASYN
    # samplers.append(('ADASYN', ADASYN(), {
    #     'n_neighbors':[5,10]
    # }))

    return models, samplers
