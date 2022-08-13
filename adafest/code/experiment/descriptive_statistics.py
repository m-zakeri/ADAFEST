"""
The module provides descriptive analysis on testability and Coverageability datasets

"""

import sys
import os
import datetime

import numpy
import scipy.stats
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler

import pandas as pd


class DescriptiveStatistic:
    def __init__(self, path=None):
        self.data_frame = pd.read_csv(path, delimiter=',', index_col=False, )
        self.data_frame.columns = [column.replace(' ', '_') for column in self.data_frame.columns]
        # self.data_frame.drop(columns=['CSORD_NumberOfClassInItsFile'], inplace=True)

    def compute_central_tendency_and_variability_measures(self):
        """
        e.g. mean, median, mode + range, variance, sd
        :return:
        """
        pass

    def compute_maximum_correlation(self, path: str = None):
        """
        https://www.simplypsychology.org/p-value.html
        :param path:
        :return:
        """
        X = self.data_frame.iloc[:, 1:]  # independent columns
        y = self.data_frame.iloc[:, -1]  # target column i.e Label_BranchCoverage

        self.data_frame['MeanCoverage'] = self.data_frame['Label_Combine1'] * 0.01
        self.data_frame['StatementCoverage'] = self.data_frame['Label_LineCoverage'] * 0.01
        self.data_frame['BranchCoverage'] = self.data_frame['Label_BranchCoverage'] * 0.01
        self.data_frame['Testability'] = self.data_frame['Coverageability1'] * 0.01

        label_coverageability = self.data_frame['MeanCoverage'] / self.data_frame['Tests']  # (Arithmetic mean)
        self.data_frame['Coverageability'] = label_coverageability

        y = label_coverageability

        # a = self.data_frame['CSORD_CountDeclClassMethod']
        # b = self.data_frame['Label_BranchCoverage']
        # print(scipy.stats.pearsonr(a, b))
        # print(scipy.stats.spearmanr(a, b))
        # r, p = scipy.stats.spearmanr(a, b)
        # print(r, p)

        # Standardization
        self.scaler = RobustScaler(with_centering=True, with_scaling=True)
        # self.scaler = StandardScaler()
        X1 = self.scaler.fit_transform(X)

        correlation_list = list()
        for col in X.columns:
            r, p = scipy.stats.pearsonr(X1[col], y)
            r2, p2 = scipy.stats.spearmanr(X1[col], y)
            correlation_list.append([col, round(r, 5), round(p, 5), round(r2, 5), round(p2, 5), ])
        df = pd.DataFrame(correlation_list,
                          columns=['Metric', 'Pearsonr', 'PearsonrPvalue', 'Spearmanr', 'SpearmanrPvalue'])
        df = df.sort_values(by=['Pearsonr'], ascending=False)
        print(df)
        df.to_csv('dataset06/Pearsonr_Coverageability_paper3.csv', index=False)

    def compute_linear_regression(self):
        """
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.linregress.html
        :return:
        """
        X = self.data_frame.iloc[:, 1:-4]  # independent columns
        y = self.data_frame.iloc[:, -4]  # target column i.e Label_BranchCoverage

        regression_result_list = list()
        for col in X.columns:
            slope, intercept, rvalue, pvalue, stderr = scipy.stats.linregress(x=self.data_frame[col], y=y)
            regression_result_list.append([col, round(slope, 5),
                                           round(intercept, 5),
                                           round(rvalue, 5),
                                           round(pvalue, 5),
                                           round(stderr, 5)])

        df = pd.DataFrame(regression_result_list,
                          columns=['metric', 'slope', 'intercept', 'rvalue', 'pvalue', 'stderr'])
        df = df.sort_values(by=['rvalue'], ascending=False)
        print(df)
        print(df.shape)
        df.to_csv('dataset05/linear_least-squares_regression_results.csv', index=False)


def main():
    DS01_path = r'dataset03/DS03014.csv'
    DS05_path = r'dataset05/DS05022.csv'
    DS05_path_pca = r'dataset05/DS05423_trainX6.csv'
    DS06_path = r'../../dataset06/DS06013.csv'

    statistic = DescriptiveStatistic(path=DS06_path, )
    statistic.compute_maximum_correlation()
    # statistic.compute_linear_regression()


if __name__ == '__main__':
    main()
