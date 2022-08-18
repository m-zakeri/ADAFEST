"""


"""

import pandas as pd
import matplotlib.pyplot as plt
from joblib import load
from matplotlib.colors import ListedColormap

import seaborn as sns

sns.set()
# import pandas.rpy.common as com

import scipy.stats
import numpy as np
from scipy.stats import spearmanr
from scipy.cluster import hierarchy

from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from adafest.code.testability import ml_models_testability, ml_models_coverageability
from adafest.code.metrics import metrics_names


def fancy_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = hierarchy.dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        # plt.title('Hierarchical Clustering Dendrogram (truncated)')
        # plt.xlabel('sample index or (cluster size)')
        # plt.ylabel('distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata


class Visualization(object):
    def __init__(self, path=None):
        self.data_frame = pd.read_csv(path,
                                      delimiter=',',
                                      index_col=False,
                                      # usecols=[0,2]
                                      )
        self.data_frame.columns = [column.replace(' ', '_') for column in self.data_frame.columns]

    def draw_bar_chart(self, path):
        """
        Figure 1 in article
        :param path:
        :return:
        """
        labels = ['VeryLow', 'Low', 'Mean', 'High', 'VeryHigh']
        fig, ax = plt.subplots()
        self.data_frame['CoverageabilityNominal'].value_counts().reindex(
            index=[labels[0], labels[1], labels[2], labels[3], labels[4]]).plot(
            ax=ax, kind='bar')  # kind='barh'
        y = self.data_frame['CoverageabilityNominal'].value_counts().reindex(
            index=[labels[0], labels[1], labels[2], labels[3], labels[4]])
        # print(type(y))
        print('y_values: {0}'.format(y.values))
        print('number of total instances {0}:'.format(len(self.data_frame.index)))

        # plt.bar(y)
        # for index, value in enumerate(y.values):
        #     plt.text(value, index, str(value))

        for p in ax.patches:
            # ax.annotate(str(p.get_height()), (p.get_x() * 1.010, p.get_height() * 1.010)) # For print pure values
            ax.annotate(str(round(p.get_height() * 100 / len(self.data_frame.index), 2)) + '%',
                        (p.get_x() * 1.010, p.get_height() * 1.030))  # For print percentage

        plt.xlabel('Coverageability level', size=9)
        plt.ylabel('Number of class', size=9)
        plt.title('The frequency of each level', size=10)
        plt.xticks(rotation=0)
        plt.savefig(path + '.png')

        plt.show()

    def draw_bar_chart_3bars(self, path):
        """
        Figure 1 in article
        :param path:
        :return:
        """
        labels = ['Low', 'Moderate', 'High', ]
        fig, ax = plt.subplots()
        self.data_frame['CoverageabilityNominalCombined'].value_counts().reindex(
            index=[labels[0], labels[1], labels[2]]).plot(
            ax=ax, kind='bar')  # kind='barh'

        y = self.data_frame['CoverageabilityNominalCombined'].value_counts().reindex(
            index=[labels[0], labels[1], labels[2]])
        # print(type(y))
        print('y_values: {0}'.format(y.values))
        print('number of total instances {0}:'.format(len(self.data_frame.index)))

        # plt.bar(y)
        # for index, value in enumerate(y.values):
        #     plt.text(value, index, str(value))

        for p in ax.patches:
            # ax.annotate(str(p.get_height()), (p.get_x() * 1.010, p.get_height() * 1.010)) # For print pure values
            ax.annotate(str(round(p.get_height() * 100 / len(self.data_frame.index), 2)) + '%',
                        (p.get_x() * 1.010, p.get_height() * 1.030))  # For print percentage

        plt.xlabel('Coverageability level', size=9)
        plt.ylabel('Number of class', size=9)
        plt.title('The frequency of each level', size=10)
        plt.xticks(rotation=0)
        plt.savefig(path + '.png')

        plt.show()

    def draw_box_whisker(self, path, path2):
        df2 = pd.read_csv(path2,
                          delimiter=',',
                          index_col=False,
                          # usecols=[0,2]
                          )
        # df3 = pd.DataFrame(data=None, index=len(self.data_frame.index), columns=2)
        df3 = self.data_frame[['ClassOrdinary_AvgLineCode']]
        df3['Average LOC without outliers'] = self.data_frame[['ClassOrdinary_AvgLineCode']]
        df3['Average LOC'] = df2['ClassOrdinary_AvgLineCode']
        boxplot = df3.boxplot(column=[  # 'ClassOrdinary_CountLineCode',
            # 'ClassOrdinary_MaxNesting',
            # 'ClassOrdinary_MaxCyclomatic'
            #   'ClassOrdinary_CountDeclMethod'
            'Average LOC',
            'Average LOC without outliers',

        ])
        plt.savefig(path + '.png')
        plt.show()

    def draw_histogram_chart(self, path):
        # df = self.data_frame[['ClassOrdinary_CountDeclMethod', 'ClassOrdinary_AvgLineCode']]
        # df = self.data_frame[['ClassOrdinary_AvgCyclomatic']]
        # df = self.data_frame[['ClassOrdinary_SumCyclomaticModified']]
        df = self.data_frame[['ClassOrdinary_AvgLineCode']]

        # _, bins, _ = df.plot.hist(bins=25, alpha=0.5, density=1)
        # plt.xlabel('Testability level', size=9)
        # plt.ylabel('Number of class', size=9)

        # data = np.random.normal(0, 1, 1000)
        data = df.to_numpy()
        _, bins, _ = plt.hist(data, bins=25,
                              density=1,
                              alpha=0.5)
        plt.title('Histogram for average line of code', size=10)

        mu, sigma = scipy.stats.norm.fit(data)
        best_fit_line = scipy.stats.norm.pdf(bins, mu, sigma)
        plt.plot(bins, best_fit_line)

        # plt.show()
        plt.savefig(path + '.png')

    def draw_heatmap_of_the_correlated_features(self, path):
        # self.data_frame = self.data_frame.sample(frac=0.20, replace=False)

        X = self.data_frame.iloc[:, 1:-4]  # independent columns
        y = self.data_frame.iloc[:, -3]  # target column i.e coverageability

        X.columns = metrics_names.top20_metrics

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))

        corr = spearmanr(X).correlation
        corr_linkage = hierarchy.ward(corr)
        # corr_linkage = hierarchy.linkage(corr, method='single')
        dendro = hierarchy.dendrogram(
            corr_linkage,
            labels=X.columns,
            ax=ax2,
            leaf_rotation=90
        )
        ax2.set(xlabel='Metrics', ylabel='Threshold')
        ax2.set_title('(b)',
                      # y=-0.10
                      )

        dendro_idx = np.arange(0, len(dendro['ivl']))

        im = ax1.imshow(corr[dendro['leaves'], :][:, dendro['leaves']],
                        vmin=-1,
                        vmax=1,
                        # origin='upper'
                        )

        # create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        # divider = make_axes_locatable(ax1)
        # cax = divider.append_axes('bottom', size="5%", pad=0.05)

        # Create colorbar
        cbar = ax1.figure.colorbar(im, ax=ax1, fraction=0.045, pad=0.05)
        cbar.ax.set_ylabel('Correlation', rotation=-90, va='bottom')

        ax1.set_xticks(dendro_idx)
        ax1.set_yticks(dendro_idx)
        ax1.set_xticklabels(dendro['ivl'], rotation='vertical')
        ax1.set_yticklabels(dendro['ivl'])
        ax1.set_title('(a)',
                      y=1.05
                      )

        # plt.colorbar()
        fig.tight_layout()
        plt.subplots_adjust(wspace=0.50)
        plt.savefig(r'charts/heatmap_of_the_correlated_features_v1_fig3.png')
        plt.show()

    def draw_heatmap_of_the_correlated_features2(self, path):
        # self.data_frame = self.data_frame.sample(frac=0.20, replace=False)
        # Determine colimns
        # X = self.data_frame.iloc[:, 1:-4]  # independent columns
        # y = self.data_frame.iloc[:, -3]  # target column i.e coverageability

        df1 = pd.read_csv(r'../../dataset06/DS06013.csv', delimiter=',', index_col=False, )
        X = df1[metrics_names.top20_metrics]

        # X.columns = metrics_names.top20_metrics
        paper_metrics_name = list()
        for col_name_ in X.columns:
            paper_metrics_name.append(metrics_names.metric_map[col_name_])

        X.columns = paper_metrics_name

        # Draw plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.5))
        # ax2.rc('xtick', labelsize=8)
        fig.subplots_adjust(wspace=1.5)

        corr = spearmanr(X).correlation
        # corr = X.corr()
        corr_linkage = hierarchy.ward(corr)
        # corr_linkage = hierarchy.linkage(corr, method='single')
        hierarchy.set_link_color_palette(['m', 'c', 'y', 'k'])
        dendro = hierarchy.dendrogram(
            corr_linkage,
            p=20,
            truncate_mode='level',
            distance_sort='descending',
            orientation='top',
            count_sort='descending',
            show_leaf_counts=True,
            # color_threshold='m',
            above_threshold_color='y',
            show_contracted=True,
            labels=np.array(X.columns),
            ax=ax2,
            leaf_rotation=90
        )
        ax2.axhline(y=0.25, c='k')
        # ax2.axis(ymin=0, ymax=0.1)
        ax2.set(
            # xlabel='Metrics',
            ylabel='Threshold')
        ax2.set_title('(b)',
                      # y=-0.10
                      )
        # ax2.

        print(len(dendro['ivl']))
        dendro_idx = np.arange(0, len(dendro['ivl']))
        # dendro_idx = np.arange(0, 15)

        im = ax1.imshow(
            # corr,
            corr[dendro['leaves'], :][:, dendro['leaves']],
            vmin=-1,
            vmax=1,
            # origin='upper'
        )

        # create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        # divider = make_axes_locatable(ax1)
        # cax = divider.append_axes('bottom', size="5%", pad=0.05)

        # Create colorbar
        cbar = ax1.figure.colorbar(im, ax=ax1, fraction=0.045, pad=0.05)
        cbar.ax.set_ylabel('Correlation', rotation=-90, va='bottom')

        ax1.set_xticks(dendro_idx)
        ax1.set_yticks(dendro_idx)

        ax1.set_xticklabels(
            # X.columns,
            dendro['ivl'],
            rotation='vertical')
        ax1.set_yticklabels(
            # X.columns,
            dendro['ivl']
        )
        ax1.set_title('(a)',
                      # y=1.280
                      )

        # plt.colorbar()
        fig.tight_layout()
        plt.savefig(r'charts/heatmap_of_the_correlated_features_v4_fig2.png')
        plt.show()

    def draw_tree_based_feature_importance(self, path):
        # https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html#sphx-glr-auto-examples-inspection-plot-permutation-importance-multicollinear-py
        # https://christophm.github.io/interpretable-ml-book/feature-importance.html

        # X = self.data_frame.iloc[:, 1:-4]  # independent columns
        # y = self.data_frame.iloc[:, -3]  # target column i.e coverageability

        X = self.data_frame.iloc[:, :-1]  # independent columns
        y = self.data_frame.iloc[:, -1]  # target column i.e coverageability

        # X.columns = metrics_names.top20_metrics

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        result = permutation_importance(clf, X_train, y_train,
                                        n_repeats=10,
                                        random_state=42)
        perm_sorted_idx = result.importances_mean.argsort()

        tree_importance_sorted_idx = np.argsort(clf.feature_importances_)
        tree_indices = np.arange(0, len(clf.feature_importances_)) + 0.5

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
        ax1.barh(tree_indices,
                 clf.feature_importances_[tree_importance_sorted_idx],
                 height=0.75)
        ax1.set_yticklabels(X.columns[tree_importance_sorted_idx])
        ax1.set_yticks(tree_indices)
        ax1.set_ylim((0, len(clf.feature_importances_)))
        ax1.set(xlabel='Importance score', ylabel='Metric')
        ax1.set_title('(a)', y=-0.10)

        ax2.boxplot(result.importances[perm_sorted_idx].T,
                    vert=False,
                    labels=X.columns[perm_sorted_idx])
        ax2.set(xlabel='Importance score', ylabel='Metric')
        ax2.set_title('(b)', y=-0.10)

        fig.tight_layout()
        plt.subplots_adjust(wspace=0.50)
        plt.savefig(r'charts/tree_based_feature_importance_v4_fig1.png')
        plt.show()

    def draw_heatmap_of_the_correlated_features_seaborn(self, path: str = None):
        testability_labels = {'VeryLow': 1, 'Low': 2, 'Mean': 3, 'High': 4, 'VeryHigh': 5}
        # self.data_frame.CoverageabilityNominal = [testability_labels[item] for item in
        #                                           self.data_frame.CoverageabilityNominal]

        # self.data_frame = self.data_frame.sample(frac=0.20, replace=False)

        X = self.data_frame.iloc[:, 1:-4]  # independent columns
        y = self.data_frame.iloc[:, -3]  # target column i.e coverageability

        X.columns = metrics_names.top20_metrics

        Xcorr = X.corr()

        self.data_frame.rename(columns={'TestabilityNominal': 'Testability', }, inplace=True)

        # sns.set_style('whitegrid')
        ax = sns.clustermap(
            data=Xcorr,
            # data=X,
            # metric='correlation',
            annot=True,
            fmt='.2g',
            # square=True,
            # xticklabels=5,
            row_cluster=True,
            col_cluster=True,
            row_colors=None,
            col_colors=None,
            # cbar_pos=None,
            # cmap='mako',
            # cmap='CMRmap',
            vmin=-1, vmax=1
        )

        plt.tight_layout()
        plt.savefig(r'charts/heatmap_of_the_correlated_features_v2_fig2.png')
        plt.show()

    def draw_dataset(self, path: str = None):
        testability_labels = {'VeryLow': 1, 'Low': 2, 'Mean': 3, 'High': 4, 'VeryHigh': 5}
        self.data_frame.CoverageabilityNominal = [testability_labels[item] for item in
                                                  self.data_frame.CoverageabilityNominal]
        X = self.data_frame.iloc[:, 1:-4]  # independent columns
        y = self.data_frame.iloc[:, -3]  # target column i.e coverageability

        # sss = StratifiedShuffleSplit(n_splits=5, test_size=0.5, random_state=0)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.99, random_state=0, stratify=y)

        x_min, x_max = X.iloc[:, 0].min() - .5, X.iloc[:, 0].max() + .5
        y_min, y_max = X.iloc[:, 1].min() - .5, X.iloc[:, 1].max() + .5
        h = .02  # step size in the mesh
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        # just plot the dataset first
        fig, ax = plt.subplots()
        # cm = plt.cm.RdBu
        cm_bright = ListedColormap(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
        # ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        ax.set_title('Coverageability Distribution')

        # Plot the training points
        scatter = ax.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1],
                             c=y_train,
                             cmap=cm_bright,
                             edgecolors='k',
                             # markers=['+', '-', '*', '^', 'o']
                             )

        # Plot the testing points
        # scatter = ax.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], c=y_test, cmap=cm_bright, alpha=0.6, edgecolors='k')

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())

        # legend1 = ax.legend(*scatter.legend_elements(),  title='Coverageability')
        legend1 = ax.legend(handles=scatter.legend_elements()[0],
                            labels=['VeryLow', 'Low', 'Mean', 'High', 'VeryHigh'],
                            title='Coverageability',
                            loc="upper right")
        ax.add_artist(legend1)

        plt.tight_layout()
        plt.show()

    # Using seaborn
    def draw_dataset2(self, path: str = None):
        testability_labels = {'VeryLow': 1, 'Low': 2, 'Mean': 3, 'High': 4, 'VeryHigh': 5}
        # self.data_frame.CoverageabilityNominal = [testability_labels[item] for item in
        #                                           self.data_frame.CoverageabilityNominal]

        self.data_frame = self.data_frame.sample(frac=0.05, replace=True)

        X = self.data_frame.iloc[:, 1:-4]  # independent columns
        y = self.data_frame.iloc[:, -3]  # target column i.e coverageability

        ax = sns.scatterplot(x='0', y='1',
                             hue='CoverageabilityNominal',
                             # hue='Label_BranchCoverage',
                             style='CoverageabilityNominal',
                             size='Label_BranchCoverage',
                             data=self.data_frame,
                             sizes=(0., 100.),
                             style_order=['VeryHigh', 'High', 'Mean', 'Low', 'VeryLow'],
                             hue_order=['VeryHigh', 'High', 'Mean', 'Low', 'VeryLow'],
                             )
        ax.set_title('Coverageability Distribution')
        ax.set_xticks(())
        ax.set_yticks(())
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.show()

    # 3D
    def draw_dataset3(self, path: str = None):
        testability_labels = {'VeryLow': 1, 'Low': 2, 'Mean': 3, 'High': 4, 'VeryHigh': 5}
        self.data_frame.CoverageabilityNominal = [testability_labels[item] for item in
                                                  self.data_frame.CoverageabilityNominal]

        self.data_frame = self.data_frame.sample(frac=0.05, replace=True)

        # X = self.data_frame.iloc[:, 1:-4]  # independent columns
        # y = self.data_frame.iloc[:, -3]  # target column i.e coverageability

        # Creating dataset
        x = self.data_frame.iloc[:, 2]
        y = self.data_frame.iloc[:, 3]
        z = self.data_frame['Label_BranchCoverage']

        # x = self.data_frame['CSORD_CountLineCodeExe']
        # z = self.data_frame['CSORD_SumCyclomatic']
        # y = self.data_frame['CSORD_CountClassCoupled']
        # y = self.data_frame['CSORD_CountStmt']

        c = self.data_frame['CoverageabilityNominal']
        # c = self.data_frame.iloc[:, 4]

        # Creating figure
        fig = plt.figure(figsize=(16, 9))
        ax = plt.axes(projection="3d")

        # Add x, y gridlines
        ax.grid(b=True, color='grey',
                linestyle='-.', linewidth=0.3,
                alpha=0.2)

        # Creating color map
        # my_cmap = plt.get_cmap('hsv')
        my_cmap = ListedColormap(['#FF1F00', '#FF8E0E', '#FFD30E', '#5EFF0E', '#009C21'])

        # Creating plot
        # https://matplotlib.org/3.3.0/api/markers_api.html

        """
        mlp colors 
        b: blue
        g: green
        r: red
        c: cyan
        m: magenta
        y: yellow
        k: black
        w: white
        """
        scatter = ax.scatter3D(x, y, z,
                               alpha=0.8,
                               c=c,
                               cmap=my_cmap,
                               edgecolors='face',
                               marker='o',
                               # markersize=15
                               )

        plt.title("Coverageability distribution")
        ax.set_xlabel('CSLOC', fontweight='bold')
        ax.set_zlabel('CSCC', fontweight='bold')
        ax.set_ylabel('NOST', fontweight='bold')

        # fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
        # print(scatter.legend_elements()[0])
        legend1 = ax.legend(handles=scatter.legend_elements()[0],
                            labels=['VeryLow', 'Low', 'Mean', 'High', 'VeryHigh'],
                            title='Coverageability',
                            loc="upper right")
        ax.add_artist(legend1)

        # show plot
        plt.show()

    # Using seaborn for r
    def draw_dataset4(self, path: str = None):
        testability_labels = {'VeryLow': 1, 'Low': 2, 'Mean': 3, 'High': 4, 'VeryHigh': 5}
        self.data_frame = self.data_frame.sample(frac=0.25, replace=True)
        self.data_frame.rename(columns={'Label_Combine1': 'Code coverage'}, inplace=True)

        sns.set_style('whitegrid')
        sns.set_style("ticks", {"xtick.major.size": 10, "ytick.major.size": 10})
        sns.set_palette("Set2", 8, .75)
        sns.color_palette("rocket_r", as_cmap=True)

        g = sns.lmplot(
            x='CSLEX_NumberOfUniqueIdentifiers',
            y='Coverageability1',
            # hue='Code coverage',
            data=self.data_frame,
            scatter=False,
            fit_reg=True,
            # robust=True,
            order=2,
            n_boot=10000,
            # x_jitter=0.05,
            x_ci='sd',
            truncate=True,
            height=4,
            aspect=1.5,
            line_kws={'color': 'blue'}
            )

        ax = sns.scatterplot(x='CSLEX_NumberOfUniqueIdentifiers',
                             y='Coverageability1',
                             # y='CoverageabilityNominal',
                             hue='Code coverage',
                             # hue='Label_BranchCoverage',
                             # style='Label_Combine1',
                             size='Tests',
                             data=self.data_frame,
                             sizes=(1, 320),
                             estimator=None,
                             palette=sns.color_palette("rocket_r", as_cmap=True),
                             # style_order=['VeryHigh', 'High', 'Mean', 'Low', 'VeryLow'],
                             # hue_order=['VeryHigh', 'High', 'Mean', 'Low', 'VeryLow'],

                             ax=g.ax
                             )

        g.set(xlim=(0, 500))
        g.set(ylim=(0, 100))
        # ax.set_title('Coverageability Distribution')
        # ax.set_xticks(())
        # ax.set_yticks(())
        ax.set(xlabel='NOIDU', ylabel='Testability')
        ax.xaxis.grid(b=True, which='both')
        sns.despine(top=True, right=True, left=False, bottom=False, offset=None, trim=False)
        # plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(r'charts/testability_per_NOIDU_DS1_v8.png')
        plt.show()

    # Using seaborn for regression plot
    def draw_dataset5(self, path: str = None):
        """
        https://seaborn.pydata.org/tutorial/regression.html
        :param path:
        :return:
        """
        testability_labels = {'VeryLow': 1, 'Low': 2, 'Mean': 3, 'High': 4, 'VeryHigh': 5}
        # self.data_frame.CoverageabilityNominal = [testability_labels[item] for item in
        #                                           self.data_frame.CoverageabilityNominal]

        self.data_frame = self.data_frame.sample(frac=0.05, replace=False)

        X = self.data_frame.iloc[:, 1:-4]  # independent columns
        y = self.data_frame.iloc[:, -3]  # target column i.e coverageability

        # self.data_frame.rename(columns={'TestabilityNominal': 'Testability', }, inplace=True)

        sns.set_style('whitegrid')
        ax = sns.lmplot(
            # x='CSLEX_NumberOfKeywords',
            x='CSLEX_NumberOfUniqueIdentifiers',
            y='Coverageability1',
            # y='CoverageabilityNominal',
            # y='TestabilityBinary',
            # hue='CoverageabilityNominal',
            # hue='Label_Combine1',
            # size=2,
            # order=3,
            ci=95,
            fit_reg=False,
            robust=True,
            # n_boot=5000,
            x_jitter=0.05,
            # x_ci=95,
            # x_estimator=np.mean,
            # col='Testability',
            # col='CoverageabilityNominal',
            # col='CSORD_MaxNesting',
            # col_wrap=5,
            # height=1.5,
            scatter=True,
            # sizes=(0., 100.),
            palette="Set1",
            # palette='plasma',
            aspect=1.2,
            # sharex=True,
            # sharey=True,
            legend=False,
            # logistic=True
            truncate=True,
            logx=True,
            x_bins=100,
            # style_order=['VeryHigh', 'High', 'Mean', 'Low', 'VeryLow'],
            # hue_order=['VeryHigh', 'High', 'Mean', 'Low', 'VeryLow'],
            # col_order=['Non-Testable', 'Testable']

            data=self.data_frame,
            scatter_kws={'s': 20, },
            line_kws={'lw': 1.25,
                      'color': 'm',
                      # 'color': '#4682b4',
                      }
        )

        # x = self.data_frame['CSORD_CountLineCodeNAMM']
        # y = self.data_frame['Label_BranchCoverage']
        # sns.jointplot(x, y, kind="reg", data=self.data_frame, ax=ax)

        # ax.set_title('Coverageability Distribution')
        # ax.set_xticks(())
        # ax.set_yticks(())
        # plt.legend(loc='upper right')
        ax.set(xlabel='NOIDU', ylabel='Testability')

        plt.tight_layout()
        plt.savefig(r'charts/testability_per_NOIDU_DS1_v4.png')
        plt.show()

    # Using seaborn for regression plot
    def draw_dataset6(self, path: str = None):
        """
        https://seaborn.pydata.org/tutorial/regression.html
        :param path:
        :return:
        """
        testability_labels = {'VeryLow': 1, 'Low': 2, 'Mean': 3, 'High': 4, 'VeryHigh': 5}
        # self.data_frame.CoverageabilityNominal = [testability_labels[item] for item in
        #                                           self.data_frame.CoverageabilityNominal]

        # self.data_frame = self.data_frame.sample(frac=0.20, replace=False)

        X = self.data_frame.iloc[:, 1:-4]  # independent columns
        y = self.data_frame.iloc[:, -3]  # target column i.e coverageability

        # self.data_frame.rename(columns={'TestabilityNominal': 'Testability', }, inplace=True)

        sns.set_style('whitegrid')
        ax = sns.lmplot(
            # x='CSLEX_NumberOfKeywords',
            x='0',
            y='Label_BranchCoverage',
            # y='CoverageabilityNominal',
            # y='TestabilityBinary',
            # hue='CoverageabilityNominal',
            # hue='Label_BranchCoverage',
            # size='Label_BranchCoverage',
            # order=2,
            # ci=None,
            fit_reg=True,
            robust=True,
            n_boot=5000,
            x_jitter=.05,
            # x_ci=95,
            x_estimator=np.mean,
            col='Testability',
            # col='CoverageabilityNominal',
            # col='CSORD_MaxNesting',
            # col_wrap=5,
            # height=1.5,

            data=self.data_frame,
            # sizes=(0., 100.),
            # style_order=['VeryHigh', 'High', 'Mean', 'Low', 'VeryLow'],
            # hue_order=['VeryHigh', 'High', 'Mean', 'Low', 'VeryLow'],
            # palette="Set1",
            palette='plasma',
            aspect=1.2,
            sharex=True,
            sharey=True,
            legend=False,
            # logistic=True
            truncate=False,
            # logx=True,
            # col_order=['Non-Testable', 'Testable']
            scatter_kws={'s': 20,
                         },
            line_kws={'lw': 1.25,
                      'color': 'm',
                      # 'color': '#4682b4',
                      }
        )

        # ax.set_title('Coverageability Distribution')
        # ax.set_xticks(())
        # ax.set_yticks(())
        # plt.legend(loc='upper right')
        ax.set(xlabel='PCA-F1', ylabel='Testability')

        plt.tight_layout()
        plt.savefig(r'charts/coverageability_regression_v3_fig1.png')
        plt.show()

    def draw_dataset_bar_chart(self, path: str = None):
        testability_labels = {'VeryLow': 1, 'Low': 2, 'Mean': 3, 'High': 4, 'VeryHigh': 5}
        # self.data_frame.CoverageabilityNominal = [testability_labels[item] for item in
        #                                           self.data_frame.CoverageabilityNominal]

        self.data_frame = self.data_frame.sample(frac=0.20, replace=False)

        X = self.data_frame.iloc[:, 1:-4]  # independent columns
        y = self.data_frame.iloc[:, -3]  # target column i.e coverageability

        self.data_frame.rename(columns={'TestabilityNominal': 'Testability', }, inplace=True)

        sns.set_style('whitegrid')
        ax = sns.barplot(
            x='CoverageabilityNominal',
            # x='CSORD_CountLineCodeNAMM',
            # y='Label_BranchCoverage',
            # y='CoverageabilityNominal',
            # y='TestabilityBinary',
            y='CSORD_CountLineCodeNAMM',

            # hue='CoverageabilityNominal',
            # hue='Testability',

            data=self.data_frame

        )
        plt.tight_layout()
        plt.savefig(r'charts/coverageability_barchart_v1_fig1.png')
        plt.show()

    def draw_dataset_cat_plot(self, path: str = None):
        testability_labels = {'VeryLow': 1, 'Low': 2, 'Mean': 3, 'High': 4, 'VeryHigh': 5}
        # self.data_frame.CoverageabilityNominal = [testability_labels[item] for item in
        #                                           self.data_frame.CoverageabilityNominal]

        self.data_frame = self.data_frame.sample(frac=0.20, replace=False)

        X = self.data_frame.iloc[:, 1:-4]  # independent columns
        y = self.data_frame.iloc[:, -3]  # target column i.e coverageability

        self.data_frame.rename(columns={'TestabilityNominal': 'Testability', }, inplace=True)

        sns.set_style('whitegrid')
        fig, axs = plt.subplots(ncols=2, figsize=(12, 6))
        ax11 = sns.boxplot(
            x='CoverageabilityNominal',
            # x='CSORD_CountLineCodeNAMM',
            # y='Label_BranchCoverage',
            # y='CoverageabilityNominal',
            # y='TestabilityBinary',
            y='CSORD_CountLineCodeNAMM',

            # hue='CoverageabilityNominal',
            # hue='Testability',
            # hue='Label_BranchCoverage',

            data=self.data_frame,
            # kind='boxen',
            # kind='box',
            # kind='point',
            # kind='bar',
            # kind='strip',
            # kind='swarm',
            # kind='violin',
            # legend=True,
            dodge=True,
            order=['VeryLow', 'Low', 'Mean', 'High', 'VeryHigh'],
            # orient='h'
            ax=axs[0]
        )

        ax22 = sns.boxplot(
            x='CoverageabilityNominal',
            # x='CSORD_CountLineCodeNAMM',
            # y='Label_BranchCoverage',
            # y='CoverageabilityNominal',
            # y='TestabilityBinary',
            # y='PK_CountDeclMethodPublic',
            # y='CSORD_MaxNesting',
            y='CSLEX_NumberOfKeywords',

            # hue='CoverageabilityNominal',
            # hue='Testability',
            # hue='Label_BranchCoverage',

            data=self.data_frame,
            # kind='boxen',
            # kind='box',
            # kind='point',
            # kind='bar',
            # kind='strip',
            # kind='swarm',
            # kind='violin',
            # legend=True,
            dodge=True,
            order=['VeryLow', 'Low', 'Mean', 'High', 'VeryHigh'],
            # orient='h'
            ax=axs[1],

        )
        axs[0].set(xlabel='Coverageability', ylabel='CSLOCNAMM')
        axs[1].set(xlabel='Coverageability', ylabel='CSNOKW')

        # plt.legend(loc='upper right')
        fig.tight_layout()
        fig.savefig(r'charts/coverageability_barchart_v1_fig2.png')
        plt.show()

    def draw_permutation_importance(self, model_path=None, model=None, ):
        reg = ml_models_coverageability.Regression(df_path=r'../../dataset06/DS06013.csv')
        if model is None:
            model = load(model_path)

        # tree_importance_sorted_idx = np.argsort(model.feature_importances_)
        # print(tree_importance_sorted_idx)
        # quit()

        result = permutation_importance(model, reg.X_test, reg.y_test,
                                        scoring='neg_mean_absolute_error',
                                        n_repeats=50,
                                        random_state=42)
        perm_sorted_idx = result.importances_mean.argsort()
        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
        print('result.importances_mean', result.importances_mean)

        result_top_features = result.importances[perm_sorted_idx].T
        print(result_top_features[:, -14:-1])
        labels_list = []
        for label in reg.X_train1.columns[perm_sorted_idx][-14:-1]:
            labels_list.append(metrics_names.metric_map[label])
        # quit()
        plt.rcParams.update({'font.size': 7})
        plt.boxplot(result_top_features[:, -14:-1], vert=False,
                    labels=labels_list)
        plt.tight_layout()
        plt.savefig(model_path[:-7] + 'permutation_importance10_14_neg_mae.png')
        # plt.show()

        plt.clf()
        labels_list = []
        for label in reg.X_train1.columns[perm_sorted_idx][-19:-1]:
            labels_list.append(metrics_names.metric_map[label])
        # quit()
        plt.rcParams.update({'font.size': 7})
        plt.boxplot(result_top_features[:, -19:-1], vert=False,
                    labels=labels_list)
        plt.tight_layout()
        plt.savefig(model_path[:-7] + 'permutation_importance10_19_neg_mae.png')
        # plt.show()
        # quit()
        # ---------------------------------------
        result = permutation_importance(model, reg.X_test, reg.y_test,
                                        scoring='r2',
                                        n_repeats=50,
                                        random_state=42,
                                        n_jobs=6)
        perm_sorted_idx = result.importances_mean.argsort()
        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
        print('result.importances_mean', result.importances_mean)

        result_top_features = result.importances[perm_sorted_idx].T
        print(result_top_features[:, -14:-1])

        plt.clf()
        labels_list = []
        for label in reg.X_train1.columns[perm_sorted_idx][-14:-1]:
            labels_list.append(metrics_names.metric_map[label])
        # quit()
        plt.rcParams.update({'font.size': 7})
        plt.boxplot(result_top_features[:, -14:-1], vert=False,
                    labels=labels_list)
        plt.tight_layout()
        plt.savefig(model_path[:-7] + 'permutation_importance10_14_r2.png')
        # plt.show()

        plt.clf()
        labels_list = []
        for label in reg.X_train1.columns[perm_sorted_idx][-19:-1]:
            labels_list.append(metrics_names.metric_map[label])
        # quit()
        plt.rcParams.update({'font.size': 7})
        plt.boxplot(result_top_features[:, -19:-1], vert=False,
                    labels=labels_list)
        plt.tight_layout()
        plt.savefig(model_path[:-7] + 'permutation_importance10_19_r2.png')
        # plt.show()

        # ---------------------------------------
        result = permutation_importance(model, reg.X_test, reg.y_test,
                                        scoring='neg_mean_squared_error',
                                        n_repeats=50,
                                        random_state=42)
        perm_sorted_idx = result.importances_mean.argsort()
        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
        print('result.importances_mean', result.importances_mean)

        result_top_features = result.importances[perm_sorted_idx].T
        print(result_top_features[:, -14:-1])

        plt.clf()
        labels_list = []
        for label in reg.X_train1.columns[perm_sorted_idx][-14:-1]:
            labels_list.append(metrics_names.metric_map[label])
        # quit()
        plt.rcParams.update({'font.size': 7})
        plt.boxplot(result_top_features[:, -14:-1], vert=False,
                    labels=labels_list)
        plt.tight_layout()
        plt.savefig(model_path[:-7] + 'permutation_importance10_14_mse.png')
        # plt.show()

    def compute_permutation_importance(self, model_path=None, model=None, n_repeats=2, scoring='r2'):
        reg = ml_models_coverageability.Regression(df_path=r'../../dataset06/DS06013.csv')
        if model is None:
            model = load(model_path)

        result = permutation_importance(model, reg.X_test, reg.y_test,
                                        scoring=scoring,
                                        n_repeats=n_repeats,
                                        random_state=42,
                                        n_jobs=4)
        perm_sorted_idx = result.importances_mean.argsort()
        result_top_features = result.importances[perm_sorted_idx].T
        print('Top five metrics:\n', result_top_features[:, -5:])
        labels_list = []
        for label in reg.X_test1.columns[perm_sorted_idx]:
            labels_list.append(metrics_names.metric_map[label])
        df1 = pd.DataFrame(data=result_top_features, columns=labels_list)
        df1.to_csv(r'dataset06/importance/VR1_DS1_sc_{0}_rep{1}.csv'.format(scoring, n_repeats), index=False)
        print('Finished.')


def main():
    csv_path = r'es_complete_dataset_all_1_0_6_without_test_93col_discretize_91col_15417_outlier_removed_11002.csv'
    csv_path2 = r'es_complete_dataset_all_1_0_6_without_test_93col_discretize_91col_15417.csv'
    csv_path3 = r'dataset03/DS03300.csv'
    csv_path4 = r'dataset03/DS03202.csv'
    csv_path5 = r'dataset03/DS03201.csv'

    DS0_path = r'dataset03/DS03014.csv'
    DS1_path = r'dataset03/DS03100.csv'
    DS4_path = r'dataset03/DS03401.csv'

    ds040 = r'dataset04/DS04031.csv'
    ds04041 = r'dataset04/DS04043.csv'

    ds01_path = r'dataset04/DS04045_train.csv'

    ds04_3label_path = r'dataset04/DS04146_train.csv'

    ds05_path = r'dataset05/DS05023.csv'

    vis = Visualization(path=r'../../dataset06/DS06013.csv')

    # vis.draw_bar_chart(path=r'charts/DS04043_coverageability.png')
    # vis.draw_bar_chart_3bars(path=r'charts/DS05023_test_coverageability.png')
    # vis.draw_box_whisker(path=r'outlier_removed.csv', path2=csv_path2)
    # vis.draw_histogram_chart(path=r'charts/'+csv_path)
    # vis.draw_tree_based_feature_importance(path=ds01_path)
    # vis.draw_heatmap_of_the_correlated_features(path=r'charts/' + csv_path)
    # vis.draw_heatmap_of_the_correlated_features_seaborn()
    # vis.draw_heatmap_of_the_correlated_features2(path=r'charts/' + csv_path)

    # vis.draw_dataset(path=r'charts/')
    # vis.draw_dataset2(path=r'charts/')
    # vis.draw_dataset3(path=r'charts/')
    vis.draw_dataset4(path=r'charts/')
    quit()
    # vis.draw_dataset5(path=r'charts/')

    # vis.draw_dataset6(path=r'charts/')
    # vis.draw_dataset_cat_plot()

    # vis.draw_permutation_importance(model_path=r'sklearn_models6/VR1_DS1.joblib')

    vis.compute_permutation_importance(model_path=r'sklearn_models6/VR1_DS1.joblib', n_repeats=10,
                                       scoring='r2')
    vis.compute_permutation_importance(model_path=r'sklearn_models6/VR1_DS1.joblib', n_repeats=10,
                                       scoring='neg_mean_absolute_error')
    vis.compute_permutation_importance(model_path=r'sklearn_models6/VR1_DS1.joblib', n_repeats=10,
                                       scoring='neg_mean_squared_error')
    vis.compute_permutation_importance(model_path=r'sklearn_models6/VR1_DS1.joblib', n_repeats=10,
                                       scoring='neg_median_absolute_error')

    vis.compute_permutation_importance(model_path=r'sklearn_models6/VR1_DS1.joblib', n_repeats=30,
                                       scoring='r2')
    vis.compute_permutation_importance(model_path=r'sklearn_models6/VR1_DS1.joblib', n_repeats=30,
                                       scoring='neg_mean_absolute_error')
    vis.compute_permutation_importance(model_path=r'sklearn_models6/VR1_DS1.joblib', n_repeats=30,
                                       scoring='neg_mean_squared_error')
    vis.compute_permutation_importance(model_path=r'sklearn_models6/VR1_DS1.joblib', n_repeats=30,
                                       scoring='neg_median_absolute_error')


# -----------------------------------------------
if __name__ == '__main__':
    main()
