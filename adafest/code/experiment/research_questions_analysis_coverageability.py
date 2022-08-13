"""
This script aimed to compare result of different models together
to answer our research questions for the article 3:
Coverageability: A formalism and measurement framework to quantify software testability

Based on the computed models with script `ml_models2`

Models reported in the paper are as follows:
model1: RFR
model2: HGBR
model3: MLPR
model4: SGDR
model5: VoR

"""

from os import listdir
from os.path import isfile, join

import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
# sns.set_style('whitegrid')

from adafest.code import metrics

result_root_directory = 'sklearn_models6c/'


def coverageability_histogram(df_path=r'dataset06/DS06013.csv', ):
    df = pd.read_csv(df_path)
    df = pd.read_csv(df_path, delimiter=',', index_col=False)

    df['MeanCoverage'] = df['Label_Combine1'] * 0.01
    df['StatementCoverage'] = df['Label_LineCoverage'] * 0.01
    df['BranchCoverage'] = df['Label_BranchCoverage'] * 0.01
    df['Testability'] = df['Coverageability1'] * 0.01

    label_coverageability = df['MeanCoverage'] / df['Tests']  # (Arithmetic mean)
    df['Coverageability'] = label_coverageability

    print('Maximum', df['Testability'].max())
    print('Maximum', df['Testability'].min())
    print('Mean', df['Testability'].mean())
    print('Standard deviation', df['Testability'].std(ddof=1))
    print('Variance', df['Testability'].var(ddof=1))
    print('Q2', df['Testability'].quantile(q=0.25))
    print('Q3', df['Testability'].quantile(q=0.75))
    # df.boxplot(column=['Testability'])

    sns.boxplot(x=df['Testability'])
    ax = sns.stripplot(x=df['Testability'], size=1)

    plt.show()

    quit()

    """
    label_coverageability2 = list()  
    label_coverageability3 = list()  
    for row in df.iterrows():
        # print(row[1][-3])
        # quit()
        label_coverageability2.append((math.sqrt(row[1][-4] * row[1][-5])) / row[1][-3])  # (Geometric mean)
        label_coverageability3.append(
            ((2 * row[1][-4] * row[1][-5]) / (row[1][-4] + row[1][-5])) / row[1][-3])  # (Harmonic mean)
    label_coverageability2 = pd.DataFrame(label_coverageability2)  # (Geometric mean)
    label_coverageability3 = pd.DataFrame(label_coverageability3)  # (Harmonic mean)
    # print(label_coverageability3)
    """

    fig, ax = plt.subplots()
    g1 = sns.histplot(df,
                      # x='Coverageability',     # To draw Coverageability distribution
                      x='Testability',  # To draw Testability distribution
                      # hue='Project',
                      # hue_order=[],
                      bins=100,
                      # kind='hist',
                      # kind='ecdf',
                      kde=True,
                      # bins=[1, 2, 3, 4, 5,6,7],
                      # ax=ax
                      legend=True,
                      # rug=True,
                      element="bars",
                      log_scale=(False, True),
                      line_kws=dict(linewidth=3),
                      ax=ax

                      )

    """
    g2 = sns.histplot(df,
        x='MeanCoverage',
        bins=100,
        # kind='hist',
        # kind='ecdf',
        kde=True,
        # bins=[1, 2, 3, 4, 5,6,7],
        # ax=ax
        legend=True,
        # rug=True,
        element="bars",
        color='r',
        log_scale=(False, True),
        # ax=ax
        )
    """

    """
    g2 = sns.histplot(df,
                      x='Tests',
                      bins=30,
                      # kind='hist',
                      # kind='ecdf',
                      kde=True,
                      # bins=[1, 2, 3, 4, 5,6,7],
                      # ax=ax
                      # legend=True,
                      # rug=True,
                      element="step",
                      ax=ax
                      )
    """

    # plt.legend(loc='upper left', bbox_to_anchor=(1,1))
    # g.despine(left=True)
    ax.set(xlabel='Value')
    plt.tight_layout()
    # ax.legend(['Coverageability', 'MeanCoverage'])
    ax.legend(['Testability'])
    plt.savefig('charts3/Testability_histogram_v1.png')
    plt.show()


def fill_table6():
    """
    Table 6. Performance of Coverageability prediction models on each dataset
    :return:
    """
    coverageability_dir = result_root_directory + 'coveragability/'
    onlyfiles = [f for f in listdir(coverageability_dir) if isfile(join(coverageability_dir, f))]
    onlyfiles = [f for f in onlyfiles if f.find('R1') != -1]
    print(onlyfiles)
    for f in onlyfiles:
        print(f)
        df = pd.read_csv(join(coverageability_dir, f), )
        try:
            print('MAE: {0}, MSE: {1}, RMSE: {2}, MSlgE: {3}, MdAE: {4}, R2: {5}'.format(
                round(df['mean_absolute_error'][0], 4),
                round(df['mean_squared_error_MSE'][0], 4),
                round(df['mean_squared_error_RMSE'][0], 4),
                round(df['mean_squared_log_error'][0], 4),
                round(df['median_absolute_error'][0], 4),
                round(df['r2_score_uniform_average'][0], 4),
            ))
        except:
            print('MAE: {0}, MSE: {1}, RMSE: {2}, MSlgE: {3}, MdAE: {4}, R2: {5}'.format(
                round(df['mean_absolute_error'][0], 4),
                round(df['mean_squared_error_MSE'][0], 4),
                round(df['mean_squared_error_RMSE'][0], 4),
                'NaN',
                round(df['median_absolute_error'][0], 4),
                round(df['r2_score_uniform_average'][0], 4),
            ))
        print('-' * 50)


def merge_all_result_csv(criteria='coveragability', model_number='4'):
    result_directory = result_root_directory + criteria + '/'
    models = ['RFR', 'HGBR', 'MLPR', 'VR']
    datasets = ['DS1', 'DS2', 'DS3', 'DS4', 'DS5']

    df = pd.DataFrame(columns=['Model', 'Dataset', 'R2-score', 'MAE'])
    for model_ in models:
        for dataset_ in datasets:
            df1 = pd.read_csv(result_directory + model_ + model_number + '_' + dataset_ + '_evaluation_metrics_R1.csv')
            df1 = df1[['r2_score_uniform_average', 'mean_absolute_error', 'mean_squared_error_MSE']]
            if model_ == 'VR':
                df1.insert(loc=0, column='Model', value='VoR')
            else:
                df1.insert(loc=0, column='Model', value=model_)
            df1.insert(loc=1, column='Dataset', value=dataset_)
            df1.columns = ['Model', 'Dataset', 'R2-score', 'MAE', 'MSE']
            df = df.append(df1, ignore_index=True)

    # print(df.info)
    print(df)
    # print(df.shape)
    # quit()
    return df


def rq2_compare_dataset_results():
    global hatch
    df = merge_all_result_csv()

    # ax = sns.barplot(x='Model', y='MAE', hue='Dataset', data=df)
    ax = sns.barplot(
        # x='Model', y='MAE', hue='Dataset',
        x='Dataset', y='MSE', hue='Model',
        data=df,
        hue_order=['MLPR', 'RFR', 'HGBR', 'VoR'],
        dodge=True,
        orient='v',
        palette=sns.color_palette('hls', 4),
        # color='none',
        edgecolor='black')
    # plt.yticks((0.2, 0.3,  0.4, 0.5, 0.8))
    num_locations = 5
    hatches = itertools.cycle(['/', '//', '\\', '\\\\', 'x', '+', 'o', '///', '-', '*', 'O', '.'])
    for i, bar in enumerate(ax.patches):
        if i % num_locations == 0:
            hatch = next(hatches)
        bar.set_hatch(hatch)

    ax.legend(loc='upper center',
              bbox_to_anchor=(0.5, 1.10),
              ncol=4, fancybox=True, shadow=False)
    plt.savefig(r'charts3/compare_model_and_datasets_MSE.png')
    # plt.show()


def draw_important_features(n_features=15):
    df = pd.read_csv(r'../../dataset06/paper3_importance/coverageability/VoR1_DS3_sc_r2_rep50.csv', )
    # df2 = df.iloc[:, -1 * n_features:].iloc[:, ::-1]
    # for i, col in enumerate(df2.columns):
    #     print(i, col)
    # quit()
    # Customize matplotlib
    # sns.set_theme(style='ticks')
    # sns.set_style('whitegrid', {'axes.facecolor': '.95'})
    plt.rcParams.update(
        {
            'text.usetex': False,
            'font.family': 'stixgeneral',
            'mathtext.fontset': 'stix',
        }
    )

    sns.set_theme(rc={  # Use mathtext, not LaTeX
        'text.usetex': False,
        # Use the Computer modern font
        # 'mathtext.rm': 'Arial',
        # 'font.family': 'serif',
        # 'font.serif': 'cmr10',
        # 'mathtext.fontset': 'cm',
        # Use ASCII minus
        'axes.unicode_minus': True,
    })
    sns.set_style('ticks', {'xtick.major.size': 0.001, 'axes.facecolor': '1.0'})
    f, ax = plt.subplots(figsize=(10, 6))
    ax.set_xscale('logit')

    sns.boxplot(x='Importance', y='Metric',
                data=pd.melt(df.iloc[:, -1 * n_features:].iloc[:, ::-1], value_name='Importance', var_name='Metric', ),
                # whis=[0, 1000],
                width=.750,
                # palette='vlag',
                linewidth=.75
                )
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=30)

    # Add in points to show each observation
    sns.stripplot(x='Importance', y='Metric',
                  data=pd.melt(df.iloc[:, -1 * n_features:].iloc[:, ::-1], value_name='Importance',
                               var_name='Metric', ),
                  size=2, color='.50', linewidth=0.25, )

    # Tweak the visual presentation
    # ax.set(ylabel="")
    ax.xaxis.grid(b=True, which='both')

    sns.despine(top=True, right=True, left=False, bottom=False, offset=None, trim=False)
    plt.tight_layout()
    plt.savefig(r'charts3/important_metrics_VoR1_DS3_sc_r2_rep50.png')
    plt.show()


def draw_refactoring_impact_on_metrics():
    df = pd.read_csv(r'../../dataset06/refactored01011.csv', )

    df_importants = pd.read_csv(r'../../dataset06/importance/VR1_DS1_sc_r2_rep30.csv')
    df_importants = df_importants.iloc[:, -1 * 30:].iloc[:, ::-1]
    inv_map = {v: k for k, v in adafest.code.metrics.metrics_names.metric_map.items()}

    heldout_columns = list()
    for col_ in df_importants.columns:
        heldout_columns.append(inv_map[col_])

    to_be_removed_cols = set(adafest.code.metrics.metrics_names.metric_map.keys()).difference(heldout_columns)
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


def rq4():
    """
    To compare Coverageability accuracy with statement and branch coverage accuracy
    :return:
    """
    global hatch

    df = pd.DataFrame(columns=['Measure/Criterion', 'Model', 'Dataset', 'R2-score', 'MAE'])
    df1 = merge_all_result_csv(criteria='coveragability', model_number='4')
    df2 = merge_all_result_csv(criteria='statement', model_number='3')
    df3 = merge_all_result_csv(criteria='branch', model_number='2')
    df4 = merge_all_result_csv(criteria='average_branch_and_statement', model_number='1')

    df1.insert(loc=0, column='Measure/Criterion', value=['Coverageability'] * 20)
    df2.insert(loc=0, column='Measure/Criterion', value=['StatementCoverage'] * 20)
    df3.insert(loc=0, column='Measure/Criterion', value=['BranchCoverage'] * 20)
    df4.insert(loc=0, column='Measure/Criterion', value=['MeanCoverage'] * 20)

    df = df.append(df1, ignore_index=True)
    df = df.append(df2, ignore_index=True)
    df = df.append(df3, ignore_index=True)
    df = df.append(df4, ignore_index=True)

    print('-' * 50)
    print(df)

    ax = sns.catplot(
        x='Model', y='R2-score',
        hue='Measure/Criterion',
        # hue='Measure/Criterion',
        col='Dataset',
        hue_order=['Coverageability', 'StatementCoverage', 'BranchCoverage', 'MeanCoverage'],
        # hue_order=['MLPR', 'RFR', 'HGBR', 'VoR'],
        col_wrap=5,
        data=df,
        dodge=True,
        # orient='v',
        kind='bar',
        ci=None,
        palette=sns.color_palette('hls', 4),
        # color='none',
        edgecolor='black',
        height=3,
        aspect=.7
    )
    num_locations = 5

    for col in range(0, 5):
        hatches = itertools.cycle(['/', '//', '\\', '\\\\', 'x', '+', 'o', '///', '-', '*', 'O', '.'])
        for i, bar in enumerate(ax.axes[col].patches):
            if i % num_locations == 0:
                hatch = next(hatches)
            bar.set_hatch(hatch)

    # plt.tight_layout()
    plt.savefig(r'charts3/compare_test_criteria_with_coverageability_R2-score_v2.png')
    plt.show()


# fill_table6()
# rq2_compare_dataset_results()
# rq4()
# draw_important_features(n_features=15)
coverageability_histogram()
