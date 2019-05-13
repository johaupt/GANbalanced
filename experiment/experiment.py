import os
os.environ["MKL_NUM_THREADS"]="1"
os.environ["NUMEXPR_NUM_THREADS"]="1"
os.environ["OMP_NUM_THREADS"]="1"
os.environ["OPENBLAS_NUM_THREADS"]="1"
os.environ["VECLIB_MAXIMUM_THREADS"]="1"

import sys
import json

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from tqdm import tqdm

# Evaluation functions
from sklearn.metrics import roc_auc_score, make_scorer
from lift.perc_lift_score import perc_lift_score

# Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Samplers
from imblearn.over_sampling import SMOTE, ADASYN, SMOTENC

sys.path.append("/home/RDC/hauptjoh.hub/utils")
sys.path.append("/home/RDC/hauptjoh.hub/GANbalanced")
from wgan.imblearn import GANbalancer
import data_loader

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--datapath', required=True, help='cifar10 | lsun | imagenet | folder | lfw ')
parser.add_argument('--dataname', required=True, help='path to dataset')
parser.add_argument('--config', required=True, help='Path to Experiment configuration file (json)')
parser.add_argument('--outfile', required=True, help='Path to output file')
parser.add_argument('--n_jobs', required=True, help='Number of cores to use', default=None)
opt = parser.parse_args()

#os.system('mkdir {0}'.format(opt.outpath))

def experiment(data_path, data_name, config_file, output_file, n_jobs):

    if data_name == "coil00":
        X, y = data_loader.load_coil00(data_path)
    elif data_name == "dmc10":
        X, y = data_loader.load_dmc10(data_path)
    elif data_name == "dmc05":
        X, y = data_loader.load_dmc05(data_path)
    else:
        raise ValueError("Data name not found in available data loaders")

    assert X.shape[0] > 0

    # Initialize index lists
    idx_cont = None
    idx_cat  = None

    if idx_cat is None:
        idx_cat = list(np.where(X.dtypes == 'category')[0])
        idx_cat = [int(x) for x in idx_cat]

    if idx_cont is None:
        idx_cont = [x for x in range(X.shape[1]) if x not in idx_cat]
        idx_cont = [int(x) for x in idx_cont]

    # Initialize embedding tuples
    categorical = None
    if idx_cat is not None:
        categorical = [(i,
                        len(X.iloc[:,i].cat.categories),
                        int(min(15., np.ceil(0.5*len(X.iloc[:,i].cat.categories))))
                       )
                        for i in idx_cat]


    # Make sure categorical variables are encoded from 0
    if np.any([idx>min(idx_cat) for idx in idx_cont]):
        raise ValueError("Variables need to be ordered [cont, cat]")


    X=X.to_numpy(dtype=np.float32)
    y=y.to_numpy(dtype=np.int32)


    ### Scorer
    scorers = {'auc':make_scorer(roc_auc_score, needs_proba=True),
              'TDLift':make_scorer(perc_lift_score, needs_proba=True, percentile=0.1)}

    # Load specific experiment configuration
    with open(config_file, "r") as fp:
        config = json.load(fp)
    ### Models
    models = []
    model_fun = {"LR":LogisticRegression(solver='liblinear'),
              "RF":RandomForestClassifier(min_samples_leaf=20)
              }

    for model_name, model_params in config["models"].items():
        models.append((model_name, model_fun[model_name], model_params))

    ### Samplers
    samplers = []
    sampler_fun = {"cGAN":GANbalancer(\
                             idx_cont=idx_cont, categorical=categorical,
                             batch_size = 128, auxiliary=True),
                    "unbalanced":None,
                    "SMOTE":SMOTENC(categorical_features=idx_cat),
                    "ADASYN":ADASYN()
    }

    for sampler_name, sampler_params in config["samplers"].items():
        samplers.append((sampler_name, sampler_fun[sampler_name], sampler_params))
        
    # Cleaner
    cleaner = TomekLinksNC(categorical_features=idx_cat, sampling_strategy='auto')

    ### Pipeline construction

    preproc_sampler = ColumnTransformer([
        ('scaler', MinMaxScaler(), idx_cont),
        ('pass',   'passthrough',  idx_cat)
    ])

    preproc_clf = ColumnTransformer([
        ('pass', 'passthrough', idx_cont),
        ('ohe',   OneHotEncoder(categories='auto', handle_unknown='ignore'),  idx_cat)
    ])


    seed = 123

    score_outer = {}

    for sampler_name, sampler, sampler_grid in tqdm(samplers):

        sampler_grid = {'sampler__'+key:item for key, item in sampler_grid.items()}

        score_inner = {}

        for model_name, model, model_grid in tqdm(models):

            pipeline = Pipeline(steps=[
                ('preproc_sampler', preproc_sampler),
                ('sampler', sampler),
                ('cleaning', cleaner),
                ('preproc_clf', preproc_clf),
                ('classifier', model)
              ])

            model_grid = {'classifier__'+key:item for key, item in model_grid.items()}
            p_grid = {**sampler_grid, **model_grid}

            inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

            clf = GridSearchCV(pipeline, param_grid= p_grid, cv=inner_cv, scoring=scorers, refit='auc',
                               return_train_score=True, iid=False,
                               n_jobs=n_jobs, pre_dispatch=n_jobs*2, verbose=1)

            score_inner[model_name] = cross_validate(clf, X=X,y=y,cv=outer_cv , scoring=scorers, return_train_score=True,
                                        return_estimator=True, verbose=1, error_score='raise')
        score_outer[sampler_name] = score_inner


    scores = pd.DataFrame([{
      'sampler':sampler_name, 'model':model_name,
        'auc':np.mean(model["test_auc"]),  'auc_sd':np.std(model["test_auc"]),
        'lift0.1':np.mean(model["test_TDLift"]),  'lift0.1_sd':np.std(model["test_TDLift"]),
    } for sampler_name, sampler in score_outer.items()
        for model_name, model in sampler.items()]
    )


    tuning_results = {sampler_name:
        {model_name:
        # vstack result DataFrame for each outer fold
            pd.concat([
                # Inner CV tuning results as DataFrame
                pd.concat([pd.DataFrame(inner_cv.cv_results_['params']).astype(str),
                           pd.DataFrame({
                               'mean_test_auc':inner_cv.cv_results_['mean_test_auc'],
                               'std_test_auc':inner_cv.cv_results_['std_test_auc'],
                               'mean_test_TDLift':inner_cv.cv_results_['mean_test_TDLift'],
                               'std_test_TDLift':inner_cv.cv_results_['std_test_TDLift']
                           })
                          ], sort=False, ignore_index=False, axis=1)
                for inner_cv in model['estimator']]).groupby(list(model['estimator'][0].cv_results_['params'][0].keys())).mean().reset_index()
                for model_name, model in sampler.items()}
              for sampler_name, sampler in score_outer.items()}

    ### Collect results
    scores = pd.DataFrame([{
          'sampler':sampler_name, 'model':model_name,
            'auc':np.mean(model["test_auc"]),  'auc_sd':np.std(model["test_auc"]),
            'lift0.1':np.mean(model["test_TDLift"]),  'lift0.1_sd':np.std(model["test_TDLift"]),
        } for sampler_name, sampler in score_outer.items()
            for model_name, model in sampler.items()]
        )

    tuning_results = {sampler_name:
        {model_name:
        # vstack result DataFrame for each outer fold
            pd.concat([
                # Inner CV tuning results as DataFrame
                pd.concat([pd.DataFrame(inner_cv.cv_results_['params']).astype(str),
                           pd.DataFrame({
                               'mean_test_auc':inner_cv.cv_results_['mean_test_auc'],
                               'std_test_auc':inner_cv.cv_results_['std_test_auc'],
                               'mean_test_TDLift':inner_cv.cv_results_['mean_test_TDLift'],
                               'std_test_TDLift':inner_cv.cv_results_['std_test_TDLift']
                           })
                          ], sort=False, ignore_index=False, axis=1)
                for inner_cv in model['estimator']]).groupby(list(model['estimator'][0].cv_results_['params'][0].keys())).mean().reset_index()
                for model_name, model in sampler.items()}
              for sampler_name, sampler in score_outer.items()}


    ### Save results to file

    with open(output_file, "w+") as result_file:
        result_file.write(f"Experiment setting \n {config_file}\n\n")

        result_file.write("Prediction results\n")
        scores.to_csv(result_file, index=None, mode='a')

        for model_name,_,_ in models:
            result_file.write(f"{model_name}\n")
            tuning_results["cGAN"][model_name].sort_values(["sampler__n_iter"]).to_csv(result_file, index=None, mode='a')
            #result_file.write(f"{tuning_results.keys()}")
            result_file.write("\n")



if __name__=="__main__":
    experiment(data_path = opt.datapath, data_name=opt.dataname,
               config_file=opt.config, output_file=opt.outfile,
               n_jobs=int(opt.n_jobs))
