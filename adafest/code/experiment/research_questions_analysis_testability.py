"""
Evaluation result for IEEE TSE R2

Models reported in the paper are as follows:
model1: RFR
model2: HGBR
model3: MLPR
model4: SGDR
model5: VoR

"""

import itertools
import numpy as np
import pandas as pd
import scipy
from scipy.stats import ttest_ind, stats

import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
# sns.set_style('whitegrid')

import metrics.metrics_names


def merge_all_result_csv(result_directory_path, merged_result_path):
    models = ['SGDR', 'MLPR', 'DTR', 'RFR', 'HGBR', 'VR']
    # models = ['SGDR', 'MLPR', 'RFR', 'HGBR', 'VR']
    datasets = ['DS1', 'DS2', 'DS3', 'DS4', 'DS5']
    columns_single_file = ['mean_absolute_error', 'mean_squared_error_MSE', 'mean_squared_error_RMSE',
                           'median_absolute_error',
                           'r2_score_uniform_average', 'explained_variance_score_uniform_average']
    columns_merged_file = ['Model', 'Dataset', 'MAE', 'MSE', 'RMSE', 'MdAE', 'R2-score', 'EVRUA']
    df = pd.DataFrame(columns=columns_merged_file)
    for model_ in models:
        for dataset_ in datasets:
            df1 = pd.read_csv(result_directory_path + model_ + '1_' + dataset_ + '_evaluation_metrics_R1.csv')
            df1 = df1[columns_single_file]
            if model_ == 'VR':
                df1.insert(loc=0, column='Model', value='VoR')
            else:
                df1.insert(loc=0, column='Model', value=model_)
            df1.insert(loc=1, column='Dataset', value=dataset_)
            df1.columns = columns_merged_file
            df = df.append(df1, ignore_index=True)

    # print(df.info)
    print(df)
    df.to_csv(merged_result_path, index=False)


def compare_prediction_approaches():
    """
    Research question #2
    :return:
    """
    experiments_path = r'D:/Users/Morteza/OneDrive/Online2/_04_2o/o2_university/PhD/Project21/a112_testability_prediction/experiments/'
    xls = pd.ExcelFile(experiments_path + r'tse_R2_results.xlsx')
    # df_gt = pd.read_excel(xls, 'binary_for_learning')
    df = pd.read_excel(xls, 'testability_comparison')
    df['Approach'] = ['TP', 'LCP', 'BCP', 'BCP [36]']
    df2 = df.melt(id_vars=['Approach', ], var_name='Metric', value_name='Value')

    sns.set(font_scale=1.25)
    # colors = ['green', 'blue', 'brown', 'red']
    # sns.set_palette(sns.color_palette(colors))
    g = sns.catplot(data=df2,
                    x='Approach', y='Value', col='Metric',
                    kind='bar',
                    sharex=True, sharey=False, margin_titles=False,
                    height=4.00, aspect=0.85, orient='v',
                    dodge=True,
                    # palette=sns.set_palette(sns.color_palette(colors)),
                    palette=reversed(sns.color_palette('hls', 4)),
                    # color='none',
                    edgecolor='black')

    # Define some hatches
    hatches = ['/', '//', '///', '\\\\']

    # Loop over the bars
    for j in range(0, 5):
        for i, thisbar in enumerate(g.axes[0, j].patches):
            # Set a different hatch for each bar
            thisbar.set_hatch(hatches[i])
            thisbar.set_width(0.45)

    g.despine(left=True)
    plt.tight_layout()
    # plt.savefig(r'charts/compare_dataset_and_models_MAE.png')
    plt.show()


def draw_important_features(n_features=15):
    df = pd.read_csv(r'dataset07/tse_R2_importance/VoR1_DS1_sc_r2_rep100.csv')  # R2
    # df = pd.read_csv(r'dataset07/tse_R2_importance/VoR1_DS1_sc_neg_mean_absolute_error_rep30.csv')  # MAE
    # df = pd.read_csv(r'dataset07/tse_R2_importance/VoR1_DS1_sc_neg_mean_squared_error_rep30.csv')  # MSE
    # df = pd.read_csv(r'dataset07/tse_R2_importance/VoR1_DS1_sc_neg_median_absolute_error_rep30.csv')  # MdAE

    df.rename(columns={'CSNOST_AVG.1': 'CSNOSTD_AVG'}, inplace=True)
    df2 = pd.melt(df.iloc[:, -1 * n_features:].iloc[:, ::-1], id_vars=None,
                  value_name='Importance', var_name='Source code metric')

    sns.set_style('ticks', {'xtick.major.size': 0.0005, 'axes.facecolor': '1.0'})
    f, ax = plt.subplots(figsize=(9, 5))
    # ax.set_xscale('logit')
    sns.boxplot(data=df2,
                x='Importance', y='Source code metric',
                width=.700,
                linewidth=.70
                )

    # Add in points to show each observation
    sns.stripplot(data=df2,
                  x='Importance', y='Source code metric',
                  size=1.55,
                  color='.35',
                  linewidth=0.20, )

    # Tweak the visual presentation
    # ax.set(ylabel="")
    ax.xaxis.grid(b=True, which='both')
    sns.despine(top=True, right=True, left=False, bottom=False, offset=None, trim=False)
    plt.tight_layout()
    plt.show()


def metrics_testability_relationship(n_features=15):
    df = pd.read_csv('dataset07/DS07012.csv')
    df.rename(columns=metrics.metrics_names.metric_map, inplace=True)
    # print(df)
    df2 = pd.read_csv(r'dataset07/tse_R2_importance/VoR1_DS1_sc_r2_rep100.csv')  # R2
    df3 = df2.iloc[:, -1 * n_features:].iloc[:, ::-1]
    df3.rename(columns={'CSNOST_AVG.1': 'CSNOSTD_AVG'}, inplace=True)
    # print(df3.columns)
    df4 = df[df3.columns]
    df4['Testability'] = df['Testability']
    df4 = df4.fillna(0)

    df4 = df4[(np.abs(stats.zscore(df4)) < 3).all(axis=1)]
    print(df4)

    for col_ in df4.columns[:-1]:
        df4[col_] = (df4[col_] - df4[col_].min()) / (df4[col_].max() - df4[col_].min())
        df4[col_] = df4[col_] * (1 - 0.0001) + 0.0001
        # df4[col_].replace(to_replace=0, value=0.001, inplace=True)

    col_order = ['CSLOCE_AVG', 'NOCJST', 'CSLOC_AVG', 'DEPENDS', 'NOIDU', 'NODOT',
                 'CSNOIM', 'CSNOPLM', 'NIM', 'CSNOSTD_AVG', 'CSNOSM', 'NONEW', 'NOREPR',
                 'CSNOCON', 'PKNOSM']
    col_regress_info = []
    for col_ in col_order:
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(df4[col_], df4['Testability'])
        r, p = scipy.stats.pearsonr(df4[col_], df4['Testability'])
        col_regress_info.append((
            slope,
            intercept,
            r_value,
            p_value,
            std_err,
            r,
            p,
        ))

    print(col_regress_info)
    # quit()

    df4 = df4.sample(frac=0.80, ignore_index=False)
    df4 = df4.sample(frac=0.40, ignore_index=False)
    print(df4.columns)

    df4 = df4.melt(id_vars='Testability', var_name='Metric', value_name='Value', )
    print(df4)

    # sns.relplot(data=df4,
    #             x='Value', y='Testability',
    #             col='Metric', col_wrap=5,
    #             height=2.5, aspect=1.15,
    #             kind='line'
    #             )

    sns.set(font_scale=1.15)
    g = sns.FacetGrid(
        data=df4, hue='Metric', col='Metric', col_wrap=4,
        height=2.95, aspect=0.95,
        sharex=False, sharey=False,
        # palette='turbo',
        legend_out=True
    )

    g.map(
        # sns.jointplot,
        sns.regplot,
        "Value", "Testability",
        truncate=True,
        x_bins=500,
        x_ci='sd',
        ci=95,
        # scatter=False,
        n_boot=1000,
        # lw=0.5,
        line_kws={'lw': 1.5,
                  'color': 'm',
                  # 'color': '#4682b4',
                  # 'label': "y={0:.1f}x+{1:.1f}".format(2.5, 3.5)
                  },

    )

    # g.map(
    #     sns.lineplot,
    #     "Value", "Testability",
    #
    # )

    """
    g = sns.lmplot(
        data=df4,
        x='Value', y='Testability',
        hue='Metric', col='Metric', col_wrap=5,

        height=2.5, aspect=1.15,

        fit_reg=True,
        truncate=True,
        # logx=True,
        x_ci='sd',
        ci=95,
        n_boot=1000,
        # x_estimator=np.mean,
        # robust=True, order=1,

        x_bins=1000,
        # common_bins=False,

        scatter_kws={'s': 10,

                     },
        line_kws={'lw': 1.05,
                  'color': 'm',
                  # 'color': '#4682b4',
                  },
        # facet_kws=dict(sharex=False, sharey=False,)
        facet_kws={'sharey': False, 'sharex': False}

    )
    """

    # for species, ax in g.axes_dict.items():
    #     print(ax)

    i = 0
    for ax, title in zip(g.axes.flat, col_order):
        # ax.set_title(title)
        ax.text(
            0.5, 0.15, f'{round(col_regress_info[i][5], 5)}',
            fontsize=12, fontweight='bold',
            horizontalalignment='center', verticalalignment='center', transform=ax.transAxes,
            color='saddlebrown',
        )
        ax.text(
            0.5, 0.05, f'({col_regress_info[i][6]:.4E})',
            fontsize=12, fontweight='bold',
            horizontalalignment='center', verticalalignment='center', transform=ax.transAxes,
            color='saddlebrown',  # 'indigo'
        )
        i += 1

    # g.legend()
    plt.tight_layout()
    plt.show()


def compare_test_effectiveness_before_and_after_refactoring():
    """

    :return:
    """
    experiments_path = r'D:/Users/Morteza/OneDrive/Online2/_04_2o/o2_university/PhD/Project21/a112_testability_prediction/experiments/'
    xls = pd.ExcelFile(experiments_path + r'tsdd_tse_R2.xlsx')
    # df_gt = pd.read_excel(xls, 'binary_for_learning')
    df = pd.read_excel(xls, 'RQ3')

    df.drop(columns=['Class'], inplace=True)
    df2 = df.melt(id_vars=['Project', 'Stage'], var_name='Criterion', value_name='Value')
    sns.set(font_scale=1.25)
    g = sns.catplot(data=df2,
                    x='Project', y='Value', hue='Stage', col='Criterion',
                    col_wrap=5,
                    kind='box',
                    sharex=True, sharey=False, margin_titles=True,
                    height=3.75, aspect=0.95, orient='v',
                    legend_out=False, legend=False, dodge=True,
                    palette=reversed(sns.color_palette('tab10')),
                    )

    for i in range(0, 5):
        hatches = ["//", "\\", '//', '\\', '//', '\\']
        for hatch, patch in zip(hatches, g.axes[i].artists):
            patch.set_hatch(hatch)

    # g2 = sns.catplot(data=df2,
    #                  x='Project', y='Value', hue='Stage', col='Criterion',
    #                  col_wrap=5,
    #                  kind='point',
    #                  sharex=True, sharey=False, margin_titles=True,
    #                  height=2.55, aspect=1.30, orient='v',
    #                  legend_out=False, legend=False, dodge=True,
    #                  # axes=g.axes
    #                  # palette=sns.color_palette('tab10', n_colors=4),
    #                  markers=['o', 'X'], linestyles=['-', '--']
    #                  )
    g.despine(left=True)
    # g2.despine(left=True)
    # plt.legend(loc='upper center')
    g.axes[3].legend(loc='upper right', fancybox=True)
    # g2.axes[4].legend(loc='upper center')
    # plt.legend()
    plt.tight_layout()
    plt.show()


def compute_test_effectiveness():
    experiments_path = r'D:/Users/Morteza/OneDrive/Online2/_04_2o/o2_university/PhD/Project21/a112_testability_prediction/experiments/'
    xls = pd.ExcelFile(experiments_path + r'tsdd_tse_R2.xlsx')
    df = pd.read_excel(xls, 'testability')

    project_name = ['Weka', 'Scijava-common']
    df_before = df.loc[(df['Stage'] == 'Before refactoring') & (df['Project'] == project_name[1])]
    df_after = df.loc[(df['Stage'] == 'After refactoring') & (df['Project'] == project_name[1])]

    # print(df_before['Project'])
    # print('before:', df_before['Testability'].mean())
    # print('after:', df_after['Testability'].mean())
    # print('after - before:', df_after['Testability'].max() - df_before['Testability'].min())

    df_im = pd.DataFrame()
    criteria = ['Testability', 'Test effectiveness', 'Statement coverage', 'Branch coverage', 'Mutation coverage']

    for var in criteria:
        df_im['Before'] = df_before[var].values
        df_im['After'] = df_after[var].values
        df_im['Changes'] = df_im['After'] - df_im['Before'] - df_im['Before']

        mean_absolute_improvement = df_im['After'].mean() - df_im['Before'].mean()
        mean_relative_improvement = ((df_im['After'].mean() - df_im['Before'].mean()) / df_im['Before'].mean()) * 100

        # print(df_im)
        # print('Mean "{}": {}'.format(var, df_im['Changes'].mean()))
        # print('Min "{}": {}'.format(var, df_im['Changes'].min()))
        # print('Max "{}": {}'.format(var, df_im['Changes'].max()))

        print(f'{var}, MAI={mean_absolute_improvement:.4}, MRI={mean_relative_improvement:.4}%')

        s, p = ttest_ind(df_im['Before'], df_im['After'], axis=0, equal_var=True, alternative='less')
        # print('Statistical test result "{}": s={}, p={}'.format(var, s, p))
        print()
    # Result:
    # Avg: (Weka: 0.12567281520748527 + Scijava-common: 0.24044206452673073)
    # Min: 0.0873 +
    # Max:
    #


def compare_source_code_metrics_before_and_after_refactoring():
    """
    This function uses visualization techniques to compare source code metrics before and after refactoring
    :return:
    """
    experiments_path = r'D:/Users/Morteza/OneDrive/Online2/_04_2o/o2_university/PhD/Project21/a112_testability_prediction/experiments/'
    xls = pd.ExcelFile(experiments_path + r'source_code_metrics_tse_R2.xlsx')
    # df_gt = pd.read_excel(xls, 'binary_for_learning')
    df = pd.read_excel(xls, 'scm_after_before2')

    df2 = pd.read_csv(r'dataset07/tse_R2_importance/VoR1_DS1_sc_r2_rep100.csv')  # R2
    df3 = df2.iloc[:, -1 * 15:].iloc[:, ::-1]
    df3.rename(columns={'CSNOST_AVG.1': 'CSNOSTD_AVG'}, inplace=True)

    df4 = df[df3.columns]
    df4['Project'] = df['Project']
    df4['Stage'] = df['Stage']

    df5 = df4.melt(id_vars=['Project', 'Stage'], var_name='Metric', value_name='Value')

    # df5.rename(columns={'Before refactoring': 'Before', 'After refactoring': 'After'}, inplace=True)
    print(df4)
    print(df5)

    sns.set(font_scale=1.35)

    g2 = sns.catplot(data=df5,
                     x='Stage', y='Value', hue='Project', col='Metric',
                     col_wrap=5, order=['Before', 'After'],
                     kind='point',
                     sharex=True, sharey=False, margin_titles=True,
                     height=3.5, aspect=0.90, orient='v',
                     legend_out=False, legend=False, dodge=True,
                     # axes=g.axes
                     palette=reversed(sns.color_palette('tab10', n_colors=2)),
                     markers=['*', 'o'], linestyles=['dotted', 'dashed']
                     )
    # g2.set(yscale="log")
    g2.despine(left=True)
    g2.axes[14].legend(loc='upper right')
    plt.tight_layout()
    # plt.show()

    # font = {'family': 'normal', 'weight': 'normal', 'size': 18}
    # plt.rc('font', **font)
    plt.savefig('top_metrics_before_and_after_refactoring_v5.pdf')


def draw_refactoring_impact_on_metrics():
    df = pd.read_csv(r'dataset06/refactored01011.csv', )
    df_importants = pd.read_csv(r'dataset06/importance/VR1_DS1_sc_r2_rep30.csv')
    df_importants = df_importants.iloc[:, -1 * 30:].iloc[:, ::-1]
    inv_map = {v: k for k, v in metrics.metrics_names.metric_map.items()}

    heldout_columns = list()
    for col_ in df_importants.columns:
        heldout_columns.append(inv_map[col_])

    to_be_removed_cols = set(metrics.metrics_names.metric_map.keys()).difference(heldout_columns)
    to_be_removed_cols = to_be_removed_cols.difference(
        set(['Class', 'Tests', 'Label_BranchCoverage', 'Coverageability1', 'Label_Combine1', 'Label_LineCoverage']))
    df.drop(columns=to_be_removed_cols, inplace=True)

    # df.drop(columns=['Class', 'Mood'], inplace=True)
    # df = df.set_index('Mood')
    df = df.drop(columns=['CSORD_SumCountPath', 'CSORD_MaxCountPath', 'CSORD_SDCountPath', 'CSORD_AvgCountPath'])
    df2 = pd.melt(df, id_vars=['ID', 'Class', 'Version'])
    print(df2)
    df2['value'] = np.log1p(df2['value'])
    # quit()
    print(df2['value'].max())
    sns.set_style('whitegrid')
    sns.set_style('ticks', )
    kind = ['strip', 'swarm', 'box', 'violin', 'boxen', 'point', 'bar', 'count']
    """
    ax = sns.catplot(x='Version', y='value',
                     # hue='Mood',
                     dodge=True,
                     col='Class', col_wrap=4,
                     order=['Original', 'Refactored', ],
                     col_order=['C1', 'C2', 'C3', 'C4'],
                     height=4, aspect=0.5,
                     data=df2,
                     kind=[kind[6], kind[7]],
                     # s=3.50,
                     # color='0.1',
                     # marker='*',
                     palette=sns.color_palette('tab10'),
                     capsize=0.15
                     )
    """
    df2['Metric value (log)'] = df2['value']
    ax = sns.pointplot(x='Class', y='Metric value (log)',
                       hue='Version',
                       dodge=True,
                       # col='Class', col_wrap=4,
                       hue_order=['Original', 'Refactored', ],
                       # col_order=['C1', 'C2', 'C3', 'C4'],
                       # height=4,
                       # aspect=0.5,
                       data=df2,
                       # s=3.50,
                       # color='0.1',
                       # marker='*',
                       palette=sns.color_palette('tab10', n_colors=2),
                       # capsize=0.15,
                       # fliersize=3.75,
                       markers=["o", "x"],
                       linestyles=["-", "--"]
                       )

    ax = sns.stripplot(x='Class', y='Metric value (log)',
                       hue='Version',
                       hue_order=['Original', 'Refactored', ],
                       dodge=True,
                       data=df2,
                       s=5.0,
                       # color='0.1',
                       # marker='*',
                       palette=sns.color_palette('tab10'),
                       linewidth=1.75,
                       edgecolor='gray',
                       )

    # Get the handles and labels. For this example it'll be 2 tuples
    # of length 4 each.
    handles, labels = ax.get_legend_handles_labels()

    # When creating the legend, only use the first two elements
    # to effectively remove the last two.
    plt.legend(loc='upper left')
    l = plt.legend(handles[0:2], labels[0:2],
                   # bbox_to_anchor=(1, 1),
                   # loc=1,
                   # borderaxespad=0.
                   )
    ax.xaxis.grid(b=True, which='both')
    sns.despine(top=True, right=True, left=False, bottom=False, offset=None, trim=False)
    # plt.get_legend().remove()
    plt.savefig(r'charts/compare_important_metrics_sd_by_refactoring_v7.png')
    plt.show()


def draw_qmood():
    experiments_path = r'D:/Users/Morteza/OneDrive/Online2/_04_2o/o2_university/PhD/Project21/a112_testability_prediction/experiments/'
    xls = pd.ExcelFile(experiments_path + r'quality_metrics_tse_R2.xlsx')
    df = pd.read_excel(xls, 'qmood-seaborn')
    df.drop(columns=['Flexibility', 'Understandability', 'Effectiveness'], inplace=True)
    df2 = df.melt(id_vars=['Project', ], var_name='Quality attribute', value_name='Improvement')

    sns.set(font_scale=1.25)

    g = sns.catplot(data=df2,
                    x='Project', y='Improvement', col='Quality attribute',
                    kind='bar',
                    sharex=True, sharey=False, margin_titles=True,
                    height=4, aspect=0.85, orient='v',
                    legend_out=False, legend=True, dodge=True,
                    palette=reversed(sns.color_palette('tab10', n_colors=2)),
                    )
    # Define some hatches
    hatches = ['/', '\\', ]
    # Loop over the bars
    for j in range(0, 5):
        for i, thisbar in enumerate(g.axes[0, j].patches):
            # Set a different hatch for each bar
            thisbar.set_hatch(hatches[i])
            thisbar.set_width(0.35)

    g.despine(left=True)
    # plt.legend(loc='upper center')
    # g.axes[0].legend(loc='upper center')
    plt.tight_layout()
    plt.show()


# merge_all_result_csv(result_directory_path=r'sklearn_models7/', merged_result_path=r'sklearn_models7/sklearn_models7_results.csv')
compare_prediction_approaches()
# draw_important_features()
# metrics_testability_relationship()
# compare_source_code_metrics_before_and_after_refactoring()
# compare_test_effectiveness_before_and_after_refactoring()
# draw_qmood()
# compute_test_effectiveness()
