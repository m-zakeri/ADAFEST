"""
The replication package for the paper
'Learning to predict test effectiveness'
published in International Journal of Intelligent Systems.

## Goal
This script implements machine learning models
for predicting the expected value of statement and branch coverage
presented in International Journal of Intelligent Systems.


## Machine learning models
* Model 1: DecisionTreeRegressor
* Model 2: RandomForestRegressor
* Model 3: GradientBoostingRegressor
* Model 4: HistGradientBoostingRegressor
* Model 5: SGDRegressor
* Model 6: MLPRegressor


## Learning datasets
Dataset	Applied preprocessing   Number of metrics

* DS1: (default)    Simple classes elimination, data classes elimination, outliers elimination,
and metric standardization  262

* DS2:    DS1 + Feature selection   20

* DS3:    DS1 + Context vector elimination  194

* DS4:    DS1 + Context vector elimination and lexical metrics elimination  177

* DS5:    DS1 + Systematically generated metrics elimination  71

* DS6:    Top 15 important source code metrics affecting Coverageability


## Model dependent variable
E[C] = (1/2*Statement coverage + 1/2*Branch coverage) * b/|n|


## Results
The results will be saved in sklearn_models6c


## Inferences
Use the method `inference_model2` of the class `Regression` to predict testability of new Java classes

"""

__version__ = '1.1.0'
__author__ = 'Morteza Zakeri-Nasrabadi'

import math
import datetime

import joblib
import matplotlib.pyplot as plt

import pandas as pd
from joblib import dump, load

from sklearn.cluster import KMeans
from sklearn.inspection import permutation_importance
from sklearn.neural_network import MLPRegressor
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit, GridSearchCV
from sklearn import tree, preprocessing
from sklearn.metrics import *
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    HistGradientBoostingRegressor,
    VotingRegressor)

from adafest.code.metrics import metrics_names


class MultioutputClassification:
    """

    https://scikit-learn.org/stable/modules/multiclass.html#multioutput-classification
    Multioutput-multiclass classification (also known as multitask classification)
    """
    pass


class ClassificationWithClustring:
    def __init__(self, df_train_path=None, df_test_path=None):
        self.df_train = pd.read_csv(df_train_path, delimiter=',', index_col=False)
        self.df_train.columns = [column.replace(' ', '_') for column in self.df_train.columns]

        self.df_test = pd.read_csv(df_test_path, delimiter=',', index_col=False)
        self.df_test.columns = [column.replace(' ', '_') for column in self.df_test.columns]

        self.X_train = self.df_train.iloc[:, :-1]
        self.y_train = self.df_train.iloc[:, -1]

        self.X_test = self.df_test.iloc[:, :-1]
        self.y_test = self.df_test.iloc[:, -1]

    def cluster_with_kmeans(self):
        kmeans = KMeans(n_clusters=3, random_state=0, max_iter=1000)
        y_km = kmeans.fit_predict(self.X_train)

        # sc = SpectralClustering(3, affinity='precomputed', n_init=100, assign_labels='discretize', random_state=0)
        # y_km = sc.fit_predict(self.X_train.values)

        # lbl = kmeans.labels_
        # print(lbl)

        X = self.X_train.values
        # plot the 3 clusters
        plt.scatter(
            X[y_km == 0, 1], X[y_km == 0, 2],
            s=50, c='lightgreen',
            marker='s', edgecolor='black',
            label='cluster 1'
        )

        plt.scatter(
            X[y_km == 1, 1], X[y_km == 1, 2],
            s=50, c='orange',
            marker='o', edgecolor='black',
            label='cluster 2'
        )

        plt.scatter(
            X[y_km == 2, 1], X[y_km == 2, 2],
            s=50, c='lightblue',
            marker='v', edgecolor='black',
            label='cluster 3'
        )

        # plot the centroids
        plt.scatter(
            kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            s=250, marker='*',
            c='red', edgecolor='black',
            label='centroids'
        )
        plt.legend(scatterpoints=1)
        plt.grid()

        plt.show()

        self.X_train['CoverageabilityNominal'] = self.y_train
        self.X_train['Cluster'] = y_km
        print(self.X_train)
        self.X_train.to_csv('dataset05/DS05_clusters_3.csv')

        # Filter 2
        df_c0 = self.X_train.loc[(self.X_train.Cluster == 0)]
        df_c1 = self.X_train.loc[(self.X_train.Cluster == 1)]
        df_c2 = self.X_train.loc[(self.X_train.Cluster == 2)]

        import collections
        conter1 = collections.Counter(df_c0['CoverageabilityNominal'])
        print('Cluster 1', conter1)
        conter2 = collections.Counter(df_c1['CoverageabilityNominal'])
        print('Cluster 2', conter2)
        conter3 = collections.Counter(df_c2['CoverageabilityNominal'])
        print('Cluster 3', conter3)


class Regression(object):
    def __init__(self, df_path=r'dataset06/DS06013.csv', avg_type=None):
        self.df = pd.read_csv(df_path, delimiter=',', index_col=False)

        self.df['Label_Combine1'] = self.df['Label_Combine1'] * 0.01
        self.df['Label_LineCoverage'] = self.df['Label_LineCoverage'] * 0.01
        self.df['Label_BranchCoverage'] = self.df['Label_BranchCoverage'] * 0.01
        self.df['Coverageability1'] = self.df['Coverageability1'] * 0.01

        label_coverageability = self.df['Label_Combine1'] / self.df['Tests']  # (Arithmetic mean)
        if avg_type is not None:
            label_coverageability2 = list()  # (Geometric mean)
            label_coverageability3 = list()  # (Harmonic mean)
            for row in self.df.iterrows():
                print(row[1][-3])
                label_coverageability2.append(
                    (math.sqrt(row[1][-4] * row[1][-5])) / row[1][-3]
                )  # (Geometric mean)
                label_coverageability3.append(
                    ((2 * row[1][-4] * row[1][-5]) / (row[1][-4] + row[1][-5])) / row[1][-3]
                )  # (Harmonic mean)
            label_coverageability2 = pd.DataFrame(label_coverageability2)
            label_coverageability3 = pd.DataFrame(label_coverageability3)

        # print('Before applying filter:', self.df.shape)
        # self.df = self.df.loc[(self.df.Label_BranchCoverage <= 0.50)]
        # self.df = self.df.loc[(self.df.Label_LineCoverage <= 0.50)]
        # print('After applying filter:', self.df.shape)

        # index -1: Coveragability1 (i.e., Testability)
        # index -2: E[C] = 1/2 branch * line ==> models names: XXX1_DSX
        # index -3: Test suite size
        # index -4: BranchCoverage ==> model names: XXX2_DSX
        # index -5: LineCoverage ==> model names: XXX3_DSX
        self.X_train1, self.X_test1, self.y_train, self.y_test = train_test_split(
            self.df.iloc[:, 1:-5],
            # self.df.iloc[:, -2],
            # label_coverageability,
            self.df['Label_BranchCoverage'],
            test_size=0.25,
            random_state=42,
            # stratify=self.df.iloc[:, -1]
        )

        """
        # ---------------------------------------
        # -- Feature selection (For DS2)
        selector = feature_selection.SelectKBest(feature_selection.f_regression, k=15)
        # clf = linear_model.LassoCV(eps=1e-3, n_alphas=100, normalize=True, max_iter=5000, tol=1e-4)
        # clf.fit(self.X_train1, self.y_train)
        # importance = np.abs(clf.coef_)
        # print('importance', importance)
        # clf = RandomForestRegressor()
        # selector = feature_selection.SelectFromModel(clf, prefit=False, norm_order=2, max_features=20, threshold=None)
        selector.fit(self.X_train1, self.y_train)

        # Get columns to keep and create new dataframe with only selected features
        cols = selector.get_support(indices=True)
        self.X_train1 = self.X_train1.iloc[:, cols]
        self.X_test1 = self.X_test1.iloc[:, cols]
        print('Selected columns by feature selection:', self.X_train1.columns)
        # quit()
        # -- End of feature selection
        """

        # ---------------------------------------
        # Standardization
        self.scaler = preprocessing.RobustScaler(with_centering=True, with_scaling=True)
        # self.scaler = preprocessing.StandardScaler()
        self.scaler.fit(self.X_train1)
        self.X_train = self.scaler.transform(self.X_train1)
        self.X_test = self.scaler.transform(self.X_test1)
        dump(self.scaler, 'DS06510.joblib')
        # quit()

    def inference_model(self, model=None, model_path=None):
        if model is None:
            model = joblib.load(model_path)

        y_true, y_pred = self.y_test, model.predict(self.X_test[3:4, ])
        print('X_test {0}'.format(self.X_test[3:4, ]))
        print('------')
        print('y_test or y_true {0}'.format(y_true[3:4, ]))
        print('------')
        print('y_pred by model {0}'.format(y_pred))

        y_true, y_pred = self.y_test, model.predict(self.X_test)
        df_new = pd.DataFrame(columns=self.df.columns)
        for i, row in self.y_test.iteritems():
            print('', i, row)
            df_new = df_new.append(self.df.loc[i], ignore_index=True)
        df_new['y_true'] = self.y_test.values
        df_new['y_pred'] = list(y_pred)

        df_new.to_csv(model_path[:-7] + '_inference_result.csv', index=True, index_label='Row')

    def inference_model2(self, model=None, model_path=None, predict_data_path=None):
        if model is None:
            model = joblib.load(model_path)

        df_predict_data = pd.read_csv(predict_data_path, delimiter=',', index_col=False)
        X_test1 = df_predict_data.iloc[:, 1:]
        X_test = self.scaler.transform(X_test1)
        y_pred = model.predict(X_test)

        df_new = pd.DataFrame(df_predict_data.iloc[:, 0], columns=['Class'])
        df_new['PredictedTestability'] = list(y_pred)

        print(df_new)
        # df_new.to_csv(r'dataset06/refactored01010_predicted_testability.csv', index=True, index_label='Row')

    def evaluate_model(self, model=None, model_path=None):
        # X = self.data_frame.iloc[:, 1:-4]
        # y = self.data_frame.iloc[:, -4]
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

        if model is None:
            model = joblib.load(model_path)

        y_true, y_pred = self.y_test, model.predict(self.X_test)
        # y_score = model.predict_proba(X_test)

        # Print all classifier model metrics
        print('Evaluating regressor ...')
        print('Regressor minimum prediction', min(y_pred), 'Regressor maximum prediction', max(y_pred))
        df = pd.DataFrame()
        df['r2_score_uniform_average'] = [r2_score(y_true, y_pred, multioutput='uniform_average')]
        df['r2_score_variance_weighted'] = [r2_score(y_true, y_pred, multioutput='variance_weighted')]

        df['explained_variance_score_uniform_average'] = [
            explained_variance_score(y_true, y_pred, multioutput='uniform_average')]
        df['explained_variance_score_variance_weighted'] = [
            explained_variance_score(y_true, y_pred, multioutput='variance_weighted')]

        df['mean_absolute_error'] = [mean_absolute_error(y_true, y_pred)]
        df['mean_squared_error_MSE'] = [mean_squared_error(y_true, y_pred)]
        df['mean_squared_error_RMSE'] = [mean_squared_error(y_true, y_pred, squared=False)]
        df['median_absolute_error'] = [median_absolute_error(y_true, y_pred)]

        if min(y_pred) >= 0:
            df['mean_squared_log_error'] = [mean_squared_log_error(y_true, y_pred)]

        # To handl ValueError: Mean Tweedie deviance error with power=2
        # can only be used on strictly positive y and y_pred.
        if min(y_pred > 0) and min(y_true) > 0:
            df['mean_poisson_deviance'] = [mean_poisson_deviance(y_true, y_pred, )]
            df['mean_gamma_deviance'] = [mean_gamma_deviance(y_true, y_pred, )]
        df['max_error'] = [max_error(y_true, y_pred)]

        df.to_csv(model_path[:-7] + '_evaluation_metrics_R1.csv', index=True, index_label='Row')

    def evaluate_model_class(self, model=None, model_path=None):
        if model is None:
            model = joblib.load(model_path)
        y_true, y_pred = self.y_test, model.predict(self.X_test)

        df_new = pd.DataFrame(y_true)
        df_new['y_pred'] = y_pred
        testability_labels = ['VeryLow', 'Low', 'Moderate', 'High', 'VeryHigh']
        testability_labels = ['Low', 'Moderate', 'High']
        bins = [-1.250, 0.250, 0.750, 1.250]
        # bins = 5
        df_new['y_ture_nominal'] = pd.cut(df_new.loc[:, ['Coverageability1']].T.squeeze(),
                                          bins=bins,
                                          labels=testability_labels,
                                          right=True
                                          )
        df_new['y_pred_nominal'] = pd.cut(df_new.loc[:, ['y_pred']].T.squeeze(),
                                          bins=bins,
                                          labels=testability_labels,
                                          right=True
                                          )
        print(df_new)
        # df_new.to_csv('XXXXX.csv')
        y_true = df_new['y_ture_nominal']
        y_pred = df_new['y_pred_nominal']
        y_score = y_pred

        # Print all classifier model metrics
        print('Evaluating classifier ...')
        df = pd.DataFrame()
        print(y_pred)
        try:
            df['accuracy_score'] = [accuracy_score(y_true, y_pred)]
            df['balanced_accuracy_score'] = [balanced_accuracy_score(y_true, y_pred)]

            df['precision_score_macro'] = [precision_score(y_true, y_pred, average='macro')]
            df['precision_score_micro'] = [precision_score(y_true, y_pred, average='micro')]

            df['recall_score_macro'] = [recall_score(y_true, y_pred, average='macro')]
            df['recall_score_micro'] = [recall_score(y_true, y_pred, average='micro')]

            df['f1_score_macro'] = [f1_score(y_true, y_pred, average='macro')]
            df['f1_score_micro'] = [f1_score(y_true, y_pred, average='micro')]
            df['fbeta_score_macro'] = [fbeta_score(y_true, y_pred, beta=0.5, average='macro')]
            df['fbeta_score_micro'] = [fbeta_score(y_true, y_pred, beta=0.5, average='micro')]

            # df['log_loss'] = [log_loss(y_true, y_score)]

            # df['roc_auc_score_ovr_macro'] = [roc_auc_score(y_true, y_score, multi_class='ovr', average='macro')]
            # df['roc_auc_score_ovr_micro'] = [roc_auc_score(y_true, y_score, multi_class='ovr', average='weighted')]
            # df['roc_auc_score_ovo_macro'] = [roc_auc_score(y_true, y_score, multi_class='ovo', average='macro')]
            # df['roc_auc_score_ovo_micro'] = [roc_auc_score(y_true, y_score, multi_class='ovo', average='weighted')]

            # print('roc_curve_:', roc_curve(y_true, y_score))  # multiclass format is not supported

            df.to_csv(model_path[:-7] + '_evaluation_metrics_C.csv', index=True, index_label='Row')
        except:
            raise ValueError('The prediction is out of range')

    def regress_with_decision_tree(self, model_path):
        # X = self.data_frame.iloc[:, 1:-4]
        # y = self.data_frame.iloc[:, -4]
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

        clf = tree.DecisionTreeRegressor()

        # CrossValidation iterator object:
        # https://scikit-learn.org/stable/tutorial/statistical_inference/model_selection.html
        cv = ShuffleSplit(n_splits=5, test_size=0.25, random_state=42)

        # Set the parameters to be used for tuning by cross-validation
        parameters = {'max_depth': range(1, 100, 10),
                      'criterion': ['mse', 'friedman_mse', 'mae'],
                      'min_samples_split': range(2, 20, 1)
                      }

        # Set the objectives which must be optimized during parameter tuning
        # scoring = ['r2', 'neg_mean_squared_error', 'neg_root_mean_squared_error', 'neg_mean_absolute_error',]
        scoring = ['neg_root_mean_squared_error', ]

        # Find the best model using gird-search with cross-validation
        clf = GridSearchCV(clf, param_grid=parameters, scoring=scoring, cv=cv, n_jobs=4,
                           refit='neg_root_mean_squared_error')
        clf.fit(X=self.X_train, y=self.y_train)

        print('Writing grid search result ...')
        df = pd.DataFrame(clf.cv_results_, )
        df.to_csv(model_path[:-7] + '_grid_search_cv_results.csv', index=False)
        df = pd.DataFrame()
        print('Best parameters set found on development set:', clf.best_params_)
        df['best_parameters_development_set'] = [clf.best_params_]
        print('Best classifier score on development set:', clf.best_score_)
        df['best_score_development_set'] = [clf.best_score_]
        print('best classifier score on test set:', clf.score(self.X_test, self.y_test))
        df['best_score_test_set:'] = [clf.score(self.X_test, self.y_test)]
        df.to_csv(model_path[:-7] + '_grid_search_cv_results_best.csv', index=False)

        # Save and evaluate the best obtained model
        print('Writing evaluation result ...')
        clf = clf.best_estimator_
        y_true, y_pred = self.y_test, clf.predict(self.X_test)
        dump(clf, model_path)
        self.evaluate_model(model=clf, model_path=model_path)

        # Plots

        # tree.plot_tree(clf)
        # plt.show()

    def regress(self, model_path: str = None, model_number: int = None):
        """

        :param model_path:
        :param model_number: 1: DTR, 2: RFR, 3: GBR, 4: HGBR, 5: SGDR, 6: MLPR,
        :return:
        """
        regressor = None
        parameters = None
        if model_number == 1:
            regressor = tree.DecisionTreeRegressor(random_state=42, )
            # Set the parameters to be used for tuning by cross-validation
            parameters = {
                # 'criterion': ['mse', 'friedman_mse', 'mae'],
                'max_depth': range(3, 50, 5),
                'min_samples_split': range(2, 30, 2)
            }
        elif model_number == 2:
            regressor = RandomForestRegressor(random_state=42, )
            parameters = {
                'n_estimators': range(100, 200, 100),
                # 'criterion': ['mse', 'mae'],
                'max_depth': range(10, 50, 10),
                # 'min_samples_split': range(2, 30, 2),
                # 'max_features': ['auto', 'sqrt', 'log2']
            }
        elif model_number == 3:
            regressor = GradientBoostingRegressor(n_estimators=400, learning_rate=0.05, random_state=42, )
            parameters = {
                # 'loss': ['ls', 'lad', ],
                'max_depth': range(10, 50, 10),
                'min_samples_split': range(2, 30, 3)
            }
        elif model_number == 4:
            regressor = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=42, )
            parameters = {
                # 'loss': ['least_squares', 'least_absolute_deviation'],
                'max_depth': range(10, 50, 10),
                'min_samples_leaf': range(5, 50, 10)
            }
        elif model_number == 5:
            regressor = linear_model.SGDRegressor(early_stopping=True, n_iter_no_change=5, random_state=42, )
            parameters = {
                'loss': ['squared_loss', 'huber', 'epsilon_insensitive'],
                'penalty': ['l2', 'l1', 'elasticnet'],
                'max_iter': range(50, 1000, 50),
                'learning_rate': ['invscaling', 'optimal', 'constant', 'adaptive'],
                'eta0': [0.1, 0.01],
                'average': [32, ]
            }
        elif model_number == 6:
            regressor = MLPRegressor(random_state=42, )
            parameters = {
                'hidden_layer_sizes': [(256, 100), (512, 256, 100), ],
                'activation': ['tanh', ],
                'solver': ['adam', ],
                'max_iter': range(50, 200, 50)
            }

        if regressor is None:
            return
        if parameters is None:
            return

        # Set the objectives which must be optimized during parameter tuning
        # scoring = ['r2', 'neg_mean_squared_error', 'neg_root_mean_squared_error', 'neg_mean_absolute_error',]
        scoring = ['neg_root_mean_squared_error', ]
        # CrossValidation iterator object:
        # https://scikit-learn.org/stable/tutorial/statistical_inference/model_selection.html
        cv = ShuffleSplit(n_splits=5, test_size=0.25, random_state=42)
        # Find the best model using gird-search with cross-validation
        clf = GridSearchCV(regressor, param_grid=parameters, scoring=scoring, cv=cv, n_jobs=4,
                           refit='neg_root_mean_squared_error')
        print('fitting model number', model_number)
        clf.fit(X=self.X_train, y=self.y_train)

        print('Writing grid search result ...')
        df = pd.DataFrame(clf.cv_results_, )
        df.to_csv(model_path[:-7] + '_grid_search_cv_results.csv', index=False)
        df = pd.DataFrame()
        print('Best parameters set found on development set:', clf.best_params_)
        df['best_parameters_development_set'] = [clf.best_params_]
        print('Best classifier score on development set:', clf.best_score_)
        df['best_score_development_set'] = [clf.best_score_]
        print('best classifier score on test set:', clf.score(self.X_test, self.y_test))
        df['best_score_test_set:'] = [clf.score(self.X_test, self.y_test)]
        df.to_csv(model_path[:-7] + '_grid_search_cv_results_best.csv', index=False)

        # Save and evaluate the best obtained model
        print('Writing evaluation result ...')
        clf = clf.best_estimator_
        y_true, y_pred = self.y_test, clf.predict(self.X_test)
        dump(clf, model_path)

        self.evaluate_model(model=clf, model_path=model_path)
        # self.evaluate_model_class(model=clf, model_path=model_path)
        # self.inference_model(model=clf, model_path=model_path)
        print('=' * 75)

    def vote(self, model_path=None, dataset_number=1):
        # Trained regressors
        reg1 = load(r'sklearn_models6c/branch/HGBR6_DS{0}.joblib'.format(dataset_number))
        reg2 = load(r'sklearn_models6c/branch/RFR6_DS{0}.joblib'.format(dataset_number))
        reg3 = load(r'sklearn_models6c/branch/MLPR6_DS{0}.joblib'.format(dataset_number))
        # reg4 = load(r'sklearn_models6/SGDR1_DS1.joblib')

        ereg = VotingRegressor([('HGBR6_DS{0}'.format(dataset_number), reg1),
                                ('RFR6_DS{0}'.format(dataset_number), reg2),
                                ('MLPR6_DS{0}'.format(dataset_number), reg3)
                                ],
                               weights=[3. / 6., 2. / 6., 1. / 6.])

        ereg.fit(self.X_train, self.y_train)
        dump(ereg, model_path)
        self.evaluate_model(model=ereg, model_path=model_path)
        try:
            self.evaluate_model_class(model=ereg, model_path=model_path)
        except:
            print('Prediction is out of the range.')


# -------------------------------------------
def compute_permutation_importance(model_path=None, model=None, n_repeats=2, scoring='r2'):
    # reg = Regression(df_path=r'dataset06/DS06013.csv')  # DS1
    reg = Regression(df_path=r'dataset06/DS06310.csv')  # DS3

    if model is None:
        model = load(model_path)

    result = permutation_importance(
        model, reg.X_test, reg.y_test,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=42,
        n_jobs=6
    )

    perm_sorted_idx = result.importances_mean.argsort()
    result_top_features = result.importances[perm_sorted_idx].T
    print('Top five metrics:\n', result_top_features[:, -5:])
    labels_list = []
    for label in reg.X_test1.columns[perm_sorted_idx]:
        labels_list.append(metrics_names.metric_map[label])
    df1 = pd.DataFrame(data=result_top_features, columns=labels_list)
    df1.to_csv(r'dataset06/paper3_importance/coverageability/VoR1_DS3_sc_{0}_rep{1}.csv'.format(scoring, n_repeats),
               index=False)
    print('Finished.')


def train_on_ds6():
    """
    To be used for predict expected value of statement and branch coverage.
    index -1: Coveragability1 (i.e., Testability)
    index -2: E[C] = 0.5*branch + 0.5*line (Arithmetic mean) ==> models names: XXX1_DSX
    index -3: Test suite size
    index -4: BranchCoverage ==> model names: XXX2_DSX
    index -5: LineCoverage ==> model names: XXX3_DSX
    index new_col1: Coverageability (Arithmetic mean) ==> model names: XXX4_DSX
    index new_col2: Coverageability2 (Geometric mean) ==> model names: XXX5_DSX
    index new_col3: Coverageability3 (Harmonic mean) ==> model names: XXX6_DSX

    Returns:


    """

    # DS1
    # reg = Regression(df_path=r'dataset06/DS06013.csv')
    # reg.regress(model_path=r'sklearn_models6c/DTR1_DS1.joblib', model_number=1)
    # reg.regress(model_path=r'sklearn_models6c/coveragability3/RFR6_DS1.joblib', model_number=2)
    # reg.regress(model_path=r'sklearn_models6c/GBR1_DS1.joblib', model_number=3)
    # reg.regress(model_path=r'sklearn_models6c/coveragability3/HGBR6_DS1.joblib', model_number=4)
    # reg.regress(model_path=r'sklearn_models6c/coveragability3/SGDR6_DS1.joblib', model_number=5)
    # reg.regress(model_path=r'sklearn_models6c/statement/MLPR3_DS1.joblib', model_number=6)
    # reg.vote(model_path=r'sklearn_models6c/statement/VR3_DS1.joblib', dataset_number=1)

    # reg.evaluate_model(model_path=r'sklearn_models6/HGBR1_DS1.joblib',)
    # reg.inference_model2(model_path=r'sklearn_models6/VR1_DS1.joblib',
    #                      predict_data_path=r'dataset06/refactored01010.csv')
    # reg.inference_model2(model_path=r'sklearn_models6/VR1_DS1.joblib',
    #                      predict_data_path=r'D:/IdeaProjects/10_water-simulator/site_1/metrics1_1.csv')
    # quit()

    # DS 1/2
    # reg.regress(model_path=r'sklearn_models6c/DTR1_DS2.joblib', model_number=1)
    # reg.regress(model_path=r'sklearn_models6c/coveragability3/RFR6_DS2.joblib', model_number=2)
    # reg.regress(model_path=r'sklearn_models6c/GBR1_DS2.joblib', model_number=3)
    # reg.regress(model_path=r'sklearn_models6c/coveragability3/HGBR6_DS2.joblib', model_number=4)
    # reg.regress(model_path=r'sklearn_models6c/coveragability3/SGDR6_DS2.joblib', model_number=5)
    # reg.regress(model_path=r'sklearn_models6c/coveragability3/MLPR6_DS2.joblib', model_number=6)
    # reg.vote(model_path=r'sklearn_models6c/coveragability3/VR6_DS2.joblib', dataset_number=2)
    # quit()

    # DS 3
    # reg = Regression(df_path=r'dataset06/DS06310.csv')
    # reg.regress(model_path=r'sklearn_models6c/DTR1_DS3.joblib', model_number=1)
    # reg.regress(model_path=r'sklearn_models6c/coveragability3/RFR6_DS3.joblib', model_number=2)
    # reg.regress(model_path=r'sklearn_models6c/GBR1_DS3.joblib', model_number=3)
    # reg.regress(model_path=r'sklearn_models6c/coveragability3/HGBR6_DS3.joblib', model_number=4)
    # reg.regress(model_path=r'sklearn_models6c/coveragability3/SGDR6_DS3.joblib', model_number=5)
    # reg.regress(model_path=r'sklearn_models6c/statement/MLPR3_DS3.joblib', model_number=6)
    # reg.vote(model_path=r'sklearn_models6c/statement/VR3_DS3.joblib', dataset_number=3)

    # DS 4
    # reg = Regression(df_path=r'dataset06/DS06410.csv')
    # reg.regress(model_path=r'sklearn_models6c/DTR1_DS4.joblib', model_number=1)
    # reg.regress(model_path=r'sklearn_models6c/coveragability3/RFR6_DS4.joblib', model_number=2)
    # reg.regress(model_path=r'sklearn_models6c/GBR1_DS4.joblib', model_number=3)
    # reg.regress(model_path=r'sklearn_models6c/coveragability3/HGBR6_DS4.joblib', model_number=4)
    # reg.regress(model_path=r'sklearn_models6c/coveragability3/SGDR6_DS4.joblib', model_number=5)
    # reg.regress(model_path=r'sklearn_models6c/statement/MLPR3_DS4.joblib', model_number=6)
    # reg.vote(model_path=r'sklearn_models6c/statement/VR3_DS4.joblib', dataset_number=4)

    # DS5
    reg = Regression(df_path=r'dataset06/DS06510.csv')
    # reg.regress(model_path=r'sklearn_models6c/branch/DTR6_DS5.joblib', model_number=1)
    reg.regress(model_path=r'sklearn_models6c/branch/RFR6_DS5.joblib', model_number=2)
    # reg.regress(model_path=r'sklearn_models6c/branch/GBR6_DS5.joblib', model_number=3)
    reg.regress(model_path=r'sklearn_models6c/branch/HGBR6_DS5.joblib', model_number=4)
    reg.regress(model_path=r'sklearn_models6c/branch/SGDR6_DS5.joblib', model_number=5)
    reg.regress(model_path=r'sklearn_models6c/branch/MLPR6_DS5.joblib', model_number=6)

    reg.vote(model_path=r'sklearn_models6c/branch/VR6_DS5.joblib', dataset_number=5)

    # quit()

    # Added for Mr. Esmaeily work
    # DS6 (important metrics)
    df_important_metrics_path = r'dataset06/DS06610.csv'
    reg = Regression(df_path=df_important_metrics_path)
    # reg.regress(model_path=r'sklearn_models6c/coveragability_arithmetic_mean/DTR6_DS6.joblib', model_number=1)
    # reg.regress(model_path=r'sklearn_models6c/coveragability_arithmetic_mean/RFR6_DS6.joblib', model_number=2)
    # reg.regress(model_path=r'sklearn_models6c/coveragability_arithmetic_mean/GBR6_DS6.joblib', model_number=3)
    # reg.regress(model_path=r'sklearn_models6c/coveragability_arithmetic_mean/HGBR6_DS6.joblib', model_number=4)
    # reg.regress(model_path=r'sklearn_models6c/coveragability_arithmetic_mean/SGDR6_DS6.joblib', model_number=5)
    # reg.regress(model_path=r'sklearn_models6c/coveragability_arithmetic_mean/MLPR6_DS6.joblib', model_number=6)
    # reg.vote(model_path=r'sklearn_models6c/coveragability_arithmetic_mean/VR6_DS6.joblib', dataset_number=6)

    model_path = r'sklearn_models6c/coveragability/VR4_DS3.joblib'
    scoring = ['r2', 'neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_median_absolute_error']
    n_repeat = [10, 20, 30, 40, 50]
    for score in scoring:
        for r in n_repeat:
            compute_permutation_importance(model_path=model_path, scoring=score, n_repeats=r, )


def create_coverageability_dataset_with_only_important_metrics():
    """
        Create DS#6 (DS06610)
        For use in Mr Esmaeili project
        Select only top 15 important Coverageability features
        :return:
    """
    df_path = r'dataset06/DS06013.csv'
    df_important_metrics_path = r'dataset06/DS06610.csv'
    df = pd.read_csv(df_path, delimiter=',', index_col=False)

    df_imp = pd.DataFrame()
    df_imp['Class'] = df['Class']  # 0
    df_imp['CSORD_SumCyclomaticStrict'] = df['CSORD_SumCyclomaticStrict']  # 1
    df_imp['CSLEX_NumberOfConditionalJumpStatements'] = df['CSLEX_NumberOfConditionalJumpStatements']  # 2
    df_imp['CSORD_LogCyclomaticStrict'] = df['CSORD_LogCyclomaticStrict']  # 3
    df_imp['CSORD_CSNOMNAMM'] = df['CSORD_CSNOMNAMM']  # 4
    df_imp['CSORD_NIM'] = df['CSORD_NIM']  # 5
    df_imp['CSORD_LogStmtDecl'] = df['CSORD_LogStmtDecl']  # 6
    df_imp['CSORD_CountDeclMethodPrivate'] = df['CSORD_CountDeclMethodPrivate']  # 7
    df_imp['CSORD_CountDeclClassMethod'] = df['CSORD_CountDeclClassMethod']  # 8
    df_imp['CSORD_NumberOfClassConstructors'] = df['CSORD_NumberOfClassConstructors']  # 9
    df_imp['CSORD_MinLineCode'] = df['CSORD_MinLineCode']  # 10
    df_imp['CSORD_SumCyclomatic'] = df['CSORD_SumCyclomatic']  # 11
    df_imp['CSLEX_NumberOfReturnAndPrintStatements'] = df['CSLEX_NumberOfReturnAndPrintStatements']  # 12
    df_imp['CSORD_MaxInheritanceTree'] = df['CSORD_MaxInheritanceTree']  # 13
    df_imp['CSLEX_NumberOfIdentifies'] = df['CSLEX_NumberOfIdentifies']  # 14
    df_imp['CSORD_CountDeclMethodPublic'] = df['CSORD_CountDeclMethodPublic']  # 15

    # Runtime metrics
    df_imp['Label_Combine1'] = df['Label_Combine1']
    df_imp['Label_LineCoverage'] = df['Label_LineCoverage']
    df_imp['Label_BranchCoverage'] = df['Label_BranchCoverage']
    df_imp['Coverageability1'] = df['Coverageability1']
    df_imp['Tests'] = df['Tests']

    df_imp.to_csv(df_important_metrics_path, index=False)


# -----------------------------------------------
def main():
    # create_coverageability_dataset_with_only_important_metrics()
    train_on_ds6()  # Last train is performed on important Coverageability metric, DS#6


if __name__ == '__main__':
    print(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'), '\t Program Start ...')
    main()
    print(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'), '\t Program End ...')
