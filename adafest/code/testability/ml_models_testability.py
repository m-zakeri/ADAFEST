"""
The replication package for the paper
'An ensemble meta-estimator to predict source code testability'
published in Applied Soft Computing Journal.

# Goal
This script implements machine learning models
for predicting testability
presented in Applied Soft Computing Journal


## Machine learning models
* Model 1: DecisionTreeRegressor
* Model 2: RandomForestRegressor
* Model 3: GradientBoostingRegressor
* Model 4: HistGradientBoostingRegressor
* Model 5: SGDRegressor
* Model 6: MLPRegressor


## Learning datasets
Dataset: Applied preprocessing: Number of metrics

* DS1:    (default)	Simple classes elimination, data classes elimination, outliers elimination, \
and metric standardization: 262 features

* DS2:    DS1 + Feature selection: 20 features

* DS3:    DS1 + Context vector elimination: 194 features

* DS4:    DS1 + Context vector elimination and lexical metrics elimination  177

* DS5:    DS1 + Systematically generated metrics elimination    71

* DS6:    Top 15 important source code metrics affecting testability


## Model dependent variable
Testability of class X: T(X) = E[C]/ (1 + omega) ^ (|n| - 1)
             where E[C] = 1/3*StatementCoverage + 1/3*BranchCoverage + 1/3*MutationCoverage


## Results
The results will be saved in `../../results` directory


## Inferences
Use the method `inference_model2` of the class `Regression` to predict testability of new Java classes.

"""

__version__ = '1.1.0'
__author__ = 'Morteza Zakeri-Nasrabadi'

import random

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, wilcoxon, mannwhitneyu

import joblib
from joblib import dump, load

from sklearn.metrics import *
from sklearn.preprocessing import QuantileTransformer

from sklearn import linear_model, feature_selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit, GridSearchCV

from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.ensemble import VotingRegressor


class Regression:
    def __init__(self, df_path: str = None, feature_selection_mode=False):
        self.df = pd.read_csv(df_path, delimiter=',', index_col=False)
        self.X_train1, self.X_test1, self.y_train, self.y_test = train_test_split(
            self.df.iloc[:, 1:-1],
            self.df.iloc[:, -1],
            test_size=0.25,
            random_state=117,
        )

        # -- Feature selection (For DS2)
        if feature_selection_mode:
            selector = feature_selection.SelectKBest(feature_selection.f_regression, k=20)
            # clf = linear_model.LassoCV(eps=1e-3, n_alphas=100, normalize=True, max_iter=5000, tol=1e-4)
            # clf.fit(self.X_train1, self.y_train)
            # importance = np.abs(clf.coef_)
            # print('importance', importance)
            # clf = RandomForestRegressor()
            # selector = feature_selection.SelectFromModel(clf, prefit=False, norm_order=2, max_features=20,)
            selector.fit(self.X_train1, self.y_train)

            # Get columns to keep and create new dataframe with only selected features
            cols = selector.get_support(indices=True)
            self.X_train1 = self.X_train1.iloc[:, cols]
            self.X_test1 = self.X_test1.iloc[:, cols]
            print('Selected columns by feature selection:', self.X_train1.columns)
            # quit()
        # --- End of feature selection

        # Standardization
        # self.scaler = preprocessing.RobustScaler(with_centering=True, with_scaling=True, unit_variance=True)
        # self.scaler = preprocessing.StandardScaler()
        self.scaler = QuantileTransformer(n_quantiles=1000, random_state=11)
        self.scaler.fit(self.X_train1)
        self.X_train = self.scaler.transform(self.X_train1)
        self.X_test = self.scaler.transform(self.X_test1)
        dump(self.scaler, df_path[:-4] + '_scaler.joblib')
        # quit()

    def regress(self, model_path: str = None, model_number: int = None):
        """

        :param model_path:
        :param model_number: 1: DTR, 2: RFR, 3: GBR, 4: HGBR, 5: SGDR, 6: MLPR,
        :return:
        """
        regressor = None
        parameters = None
        if model_number == 1:
            regressor = DecisionTreeRegressor(random_state=23, )
            # Set the parameters to be used for tuning by cross-validation
            parameters = {
                # 'criterion': ['mse', 'friedman_mse', 'mae'],
                'max_depth': range(3, 50, 5),
                'min_samples_split': range(2, 30, 2)
            }
        elif model_number == 2:
            regressor = RandomForestRegressor(random_state=19, )
            parameters = {
                'n_estimators': range(100, 200, 100),
                # 'criterion': ['mse', 'mae'],
                'max_depth': range(10, 50, 10),
                # 'min_samples_split': range(2, 30, 2),
                # 'max_features': ['auto', 'sqrt', 'log2']
            }
        elif model_number == 3:
            regressor = GradientBoostingRegressor(n_estimators=400, learning_rate=0.05, random_state=17, )
            parameters = {
                # 'loss': ['ls', 'lad', ],
                'max_depth': range(10, 50, 10),
                'min_samples_split': range(2, 30, 3)
            }
        elif model_number == 4:
            regressor = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=13, )
            parameters = {
                # 'loss': ['least_squares', 'least_absolute_deviation'],
                'max_depth': range(10, 50, 10),
                'min_samples_leaf': range(5, 50, 10)
            }
        elif model_number == 5:
            regressor = linear_model.SGDRegressor(early_stopping=True, n_iter_no_change=5, random_state=11, )
            parameters = {
                'loss': ['squared_loss', 'huber', 'epsilon_insensitive'],
                'penalty': ['l2', 'l1', 'elasticnet'],
                'max_iter': range(50, 1000, 50),
                'learning_rate': ['invscaling', 'optimal', 'constant', 'adaptive'],
                'eta0': [0.1, 0.01],
                'average': [32, ]
            }
        elif model_number == 6:
            regressor = MLPRegressor(random_state=7, )
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
        cv = ShuffleSplit(n_splits=5, test_size=0.20, random_state=101)
        # Find the best model using gird-search with cross-validation
        clf = GridSearchCV(
            regressor,
            param_grid=parameters,
            scoring=scoring,
            cv=cv,
            n_jobs=7,
            refit='neg_root_mean_squared_error'
        )

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
        reg1 = load(r'../../results/HGBR1_DS{0}.joblib'.format(dataset_number))
        reg2 = load(r'../../results/RFR1_DS{0}.joblib'.format(dataset_number))
        reg3 = load(r'../../results/MLPR1_DS{0}.joblib'.format(dataset_number))
        # reg4 = load(r'results/SGDR1_DS1.joblib')

        ereg = VotingRegressor(
            [('HGBR1_DS{0}'.format(dataset_number), reg1),
             ('RFR1_DS{0}'.format(dataset_number), reg2),
             ('MLPR1_DS{0}'.format(dataset_number), reg3)],
            weights=[3. / 6., 2. / 6., 1. / 6.]
        )

        ereg.fit(self.X_train, self.y_train)
        dump(ereg, model_path)
        try:
            self.evaluate_model(model=ereg, model_path=model_path)
        except:
            print('Prediction is out of the range.')

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

    def evaluate_model(self, model=None, model_path=None):
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

        # To handle ValueError: Mean Tweedie deviance error with power=2 can only be used on \
        # strictly positive y and y_pred.
        if min(y_pred > 0) and min(y_true) > 0:
            df['mean_poisson_deviance'] = [mean_poisson_deviance(y_true, y_pred, )]
            df['mean_gamma_deviance'] = [mean_gamma_deviance(y_true, y_pred, )]
        df['max_error'] = [max_error(y_true, y_pred)]

        df.to_csv(model_path[:-7] + '_evaluation_metrics_R1.csv', index=True, index_label='Row')

    def statistical_comparison(self, ):
        model1 = load(r'results/VR1_DS1.joblib')
        # model2 = load(r'results/RFR1_DS1.joblib')
        # model2 = load(r'results/MLPR1_DS1.joblib')
        model2 = load(r'results/HGBR1_DS1.joblib')

        y_pred1 = model1.predict(self.X_test)
        y_pred2 = model2.predict(self.X_test)

        y_true = np.array(list(self.y_test.values))
        y_pred1 = np.array(list(y_pred1))
        y_pred2 = np.array(list(y_pred2))

        output_errors1 = (abs(y_true - y_pred1))
        output_errors2 = (abs(y_true - y_pred2))

        s, p = ttest_ind(list(output_errors1), list(output_errors2), alternative="less", )
        print(f'statistic = {s}, p-value={p}')

    def statistical_comparison2(self, ):
        model1 = load(r'results/VR1_DS1.joblib')
        # model2 = load(r'results/RFR1_DS1.joblib')
        # model2 = load(r'results/MLPR1_DS1.joblib')
        # model2 = load(r'results/HGBR1_DS1.joblib')
        # model2 = load(r'results/SGDR1_DS1.joblib')
        model2 = load(r'results/DTR1_DS1.joblib')
        # model2 = load(r'results/DTR1_DS3.joblib')

        me1 = []
        me2 = []
        for i in range(100):  # 100
            x_test = []
            y_test = []
            for j in range(2000):  # 2000
                random_index = random.randint(0, len(self.X_test) - 1)
                x_test.append(list(self.X_test)[random_index])
                y_test.append(self.y_test.values[random_index])
            y_pred1 = model1.predict(x_test)
            y_pred2 = model2.predict(x_test)
            # me1.append(mean_squared_error(y_test, y_pred1))
            # me2.append(mean_squared_error(y_test, y_pred2))
            me1.append(r2_score(y_test, y_pred1))
            me2.append(r2_score(y_test, y_pred2))

        # print('me1', me1)
        # print('me2', me2)

        s, p = ttest_ind(me2, me1, alternative="less", )
        print(f'statistic t-test = {s}, p-value={p}')
        s, p = wilcoxon(me2, me1, alternative="less", )
        print(f'statistic wilcoxon = {s}, p-value={p}')
        s, p = mannwhitneyu(me2, me1, alternative="less", )
        print(f'statistic Mann-Whitney U = {s}, p-value={p}')
        print()
