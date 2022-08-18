"""
The replication package for the paper
'An ensemble meta-estimator to predict source code testability'
published in Applied Soft Computing Journal.


The main module of ADAFEST project.
ADAFEST is an abbreviation for 'a data-driven apparatus for estimating software testability'

## Goal
This script train and evaluate all testability prediction models

## Inputs
The inputs are dataset in `../data/` directory

## Results
The results will be saved in `../results` directory

"""

__version__ = '1.1.0'
__author__ = 'Morteza Zakeri-Nasrabadi'

import datetime
import pandas as pd
import joblib

from sklearn.inspection import permutation_importance

from testability.ml_models_testability import Regression
from metrics import metrics_names


def train_and_evaluate(ds_number=1):
    """
        Dataset: Applied preprocessing: Number of metrics

        DS1: (default)  Simple classes elimination, data classes elimination, \
        outliers elimination, and metric standardization: 262 features

        DS2: DS1 + Feature selection: 20 features

        DS3: DS1 + Context vector elimination: 194 features

        DS4: DS1 + Context vector elimination and lexical metrics elimination   177

        DS5: DS1 + Systematically generated metrics elimination 71

        DS6: Top 15 important source code metrics affecting testability

    """

    reg = None
    if ds_number == 0:
        return
    elif ds_number == 1:
        reg = Regression(df_path=r'../data/DS07012.csv')  # DS1
    elif ds_number == 2:
        reg = Regression(df_path=r'../data/DS07012.csv', feature_selection_mode=True)  # DS2
    elif ds_number == 3:
        reg = Regression(df_path=r'../data/DS07310.csv')  # DS3
    elif ds_number == 4:
        reg = Regression(df_path=r'../data/DS06410.csv')  # DS4
    elif ds_number == 5:
        reg = Regression(df_path=r'../data/DS07510.csv')  # DS5
    elif ds_number == 6:
        reg = Regression(df_path=r'../data/DS07610.csv')  # DS6

    if reg is None:
        return

    reg.regress(model_path=r'../results/DTR1_DS1.joblib', model_number=1)  # Model 1
    reg.regress(model_path=f'../results/RFR1_DS{ds_number}.joblib', model_number=2)  # Model 2
    # reg.regress(model_path=f'../results/GBR1_DS{ds_number}.joblib', model_number=3)  # Model 3
    reg.regress(model_path=f'../results/HGBR1_DS{ds_number}.joblib', model_number=4)  # Model 4
    # reg.regress(model_path=f'../results/SGDR1_DS{ds_number}.joblib', model_number=5)  # Model 5
    reg.regress(model_path=f'../results/MLPR1_DS{ds_number}.joblib', model_number=6)  # Model 6
    reg.vote(model_path=f'../results/VR1_DS{ds_number}.joblib', dataset_number=ds_number)  # Ensemble meta-regressor


def compute_permutation_importance(model_path=None, model=None, n_repeats=2, scoring='r2'):
    reg = Regression(df_path=r'../data/DS07012.csv')  # DS1
    if model is None:
        model = joblib.load(model_path)
    result = permutation_importance(
        model, reg.X_test, reg.y_test,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=71,
        n_jobs=7
    )
    perm_sorted_idx = result.importances_mean.argsort()
    result_top_features = result.importances[perm_sorted_idx].T
    print('Top five metrics:\n', result_top_features[:, -5:])
    labels_list = []
    for label in reg.X_test1.columns[perm_sorted_idx]:
        labels_list.append(metrics_names.metric_map[label])
    df1 = pd.DataFrame(data=result_top_features, columns=labels_list)
    df1.to_csv(r'../results/tse_R2_importance/VoR1_DS1_sc_{0}_rep{1}.csv'.format(scoring, n_repeats),
               index=False)
    print('Finished.')


def measure_feature_importance():
    """
    """
    model_path = r'../../results/VR1_DS1.joblib'
    scoring = ['r2', 'neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_median_absolute_error']
    n_repeat = [10, 20, 30, 40, 50]
    for score in scoring:
        for r in n_repeat:
            compute_permutation_importance(model_path=model_path, scoring=score, n_repeats=r, )


def main():
    for i in range(1, 7):
        train_and_evaluate(ds_number=i)  # Train and evaluate all models of all datasets
    # measure_feature_importance()


# -----------------------------------------------
if __name__ == '__main__':
    print(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'), '\t Program Start ...')
    main()
    print(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'), '\t Program End ...')
