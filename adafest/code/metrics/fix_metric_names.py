import pandas as pd

from adafest.code.preprocessing.data_preparation_evosuite_2 import TestabilityMetrics

def create_errata_file():
    DS030_path = r'dataset03/DS030.csv'
    DS030 = pd.read_csv(DS030_path, delimiter=',', index_col=False,)

    df = pd.DataFrame()
    df['WrongNames'] = DS030.columns

    correct_columns_names = []
    correct_columns_names.append('Class')
    correct_columns_names.extend(TestabilityMetrics.get_all_metrics_names())
    correct_columns_names.extend(['Label_LineCoverage','Label_BranchCoverage','Label_MutationScore'])
    df['CorrectNames'] = correct_columns_names

    df.to_csv('dataset03/errata.csv', index=True, index_label='Row')


def extract_coverage():
    DS040_path = r'dataset04/DS040.csv'
    df_DS040 = pd.read_csv(DS040_path, delimiter=',', index_col=False, )
    runtime_result_path = r'runtime_result/DS040_labels.csv'
    df = pd.DataFrame()

    df['Class'] = df_DS040['Class']
    df['LOC'] = df_DS040['CSORD_CountLineCodeExe']

    df['LineCoverage'] = df_DS040['Label_LineCoverage']

    df['LineCoveragePercent'] = (df_DS040['Label_LineCoverage'] / df_DS040['CSORD_CountLineCodeExe']) * 100
    df['BranchCoverage'] = round(df_DS040['Label_BranchCoverage'] * 100, 2)
    df['MutationScore'] = round(df_DS040['Label_MutationScore'] * 100, 2)

    df.to_csv(runtime_result_path, index=False)


def create_evosuite160coverage_csv():
    with open(r'runtime_result/evosuit160_sf110_result_html.txt', mode='r', ) as f:
        lines = f.readlines()

    df = pd.DataFrame()
    for line in lines:
        line = line[:-1]
        record = line.split('\t')
        record[1] = str(record[1][:-1])
        record[2] = str(record[2][:-1])
        record[3] = str(record[3][:-1])
        record[4] = str(record[4][:-1])
        record[5] = str(record[5])
        record[6] = str(record[6])
        df_temp = pd.DataFrame([record], columns=['Class', 'Line', 'Branch', 'Mutation', 'Output', 'Exceptions', 'Tests'])
        # print(df_temp)
        # quit()
        df = df.append(df_temp, ignore_index=True, )

    print(df)
    df.to_csv(r'runtime_result/evosuit160_sf110_result_html.csv', index=False)


def fix_ds050_coverage_info():  # runtime variables
    DS050_path = r'../../dataset06/DS060Raw.csv'
    df_DS050 = pd.read_csv(DS050_path, delimiter=',', index_col=False, )
    df_runtime = pd.read_csv(r'runtime_result/evosuit160_sf110_result_html.csv', delimiter=',', index_col=False, )

    label_combine_line_branch = list()
    label_combine_line_branch_mutation = list()
    label_max_line_and_branch = list()
    label_min_line_and_branch = list()

    for index, row in df_DS050.iterrows():
        print('--> ', index)
        # print(row)
        class_name = df_runtime.loc[df_runtime['Class'] == row['Class']].iloc[0]
        # print(class_name)
        # quit()
        df_DS050.loc[index, 'Label_LineCoverage'] = class_name['Line']
        df_DS050.loc[index, 'Label_BranchCoverage'] = class_name['Branch']
        df_DS050.loc[index, 'Label_MutationScore'] = class_name['Mutation']

        df_DS050.loc[index, 'Output'] = class_name['Output']
        df_DS050.loc[index, 'Exceptions'] = class_name['Exceptions']
        df_DS050.loc[index, 'Tests'] = class_name['Tests']

        label_combine_line_branch.append(class_name['Line'] * 0.5 + class_name['Branch'] * 0.5)
        label_combine_line_branch_mutation.append((class_name['Line'] + class_name['Branch'] + class_name['Mutation'])/3)
        label_max_line_and_branch.append(max(class_name['Line'], class_name['Branch']))
        label_min_line_and_branch.append(min(class_name['Line'], class_name['Branch']))

    df_DS050['Label_Combine1'] = label_combine_line_branch
    df_DS050['Label_Combine2'] = label_combine_line_branch_mutation
    df_DS050['Label_MaxLineAndBranch'] = label_max_line_and_branch
    df_DS050['Label_MinLineAndBranch'] = label_min_line_and_branch

    df_DS050.to_csv(r'dataset06/DS060RawLabeled.csv', index=False)


def add_coverage_ability_column():
    df = pd.read_csv(r'../../dataset06/DS06012.csv')
    coverage_ability_list = list()
    for index, row in df.iterrows():
        # print('', row['Label_Combine1'], '-->', row['Label_Combine1'] / (1+math.log10(row['Tests'])))
        # coverage_ability_list.append(row['Label_Combine1'] / (1+math.log10(row['Tests'])))
        coverageability = row['Label_Combine1']
        x = row['Tests']
        alpha = [0.0150, 0.0075, 0.0050, 0.0025, 0.0001, 0.0500]
        testability = coverageability / ((1 + alpha[0])**(x-1))
        print('', row['Class'], testability)
        coverage_ability_list.append(testability)

    df['Coverageability1'] = coverage_ability_list
    df.to_csv(r'dataset06/DS06013.csv', index=False)


def add_project_name():
    df_runtime = pd.read_csv(r'runtime_result/evosuit160_sf110_result_html.csv', delimiter=',', index_col=False, )
    with open(r'runtime_result/es_compressed_data_all_1_0_6', mode='r', encoding='utf8') as f:
        data_lines = f.readlines()

    project_mame_list = list()
    for index, row in df_runtime.iterrows():
        # print('--> ', index)
        print('Processing class "{0}"'.format(row['Class']))
        for line in data_lines:
            items = line.split(' ')
            long_class_name = items[3][1:-1]
            if long_class_name == row['Class']:
                project_mame_list.append(items[2][1:-1])
                break
        # print(project_mame_list)
        # quit()

    df_runtime.insert(loc=0, column='Project', value=project_mame_list)
    df_runtime.to_csv(r'runtime_result/evosuit160_sf110_result_html_with_project.csv', index=False)


def query():
    DS050_path = r'../../dataset06/DS06011.csv'
    df = pd.read_csv(DS050_path, delimiter=',', index_col=False, )

    """
    # Filter 1
    df1 = df.loc[
        (df.Label_LineCoverage == 0.)
        & (df.Label_BranchCoverage == 0.)
        & (df.Label_MutationScore == 0.)
        # & (df.Output == 0.)
        # & (df.Exceptions == 0.)
        & (df.Tests == 0.)
        # & (df.CSORD_SumCyclomatic == 0)
    ]
    print(df1)
    # df1.to_html('dataset05/all_zero_runtimes.html')
    # df1.to_csv('dataset05/all_zero_runtimes.csv', index=False)
    quit()
    
    # Filter 2
    df2 = df.loc[
        (df.Label_LineCoverage == 100.0)
        & (df.Label_BranchCoverage == 100.0)
        & (df.Label_MutationScore == 100.0)
        & (df.Output == 100.0)
        # & (df.Exceptions == 0.)
        & (df.Tests <= 5)
        # & (df.CSORD_SumCyclomatic == 0)
        ]
    # print(df2)
    # df2.to_csv('dataset05/all_100_runtimes.csv', index=False)

    # Filter 3
    df3 = df.loc[
        (df.CSORD_CountLineCode < 10)
        # & (df.Label_BranchCoverage < 80.0)
        ]
    # print(df3)
    # df3.to_csv('dataset05/all_100_runtimes.csv', index=False)
    """

    # Filter 4
    df4 = df.loc[(df.Label_Combine5 == 100)]
    print(df4)
    # df3.to_csv('dataset05/all_100_runtimes.csv', index=False)


def compare_class_file_metric_file():
    class_file_path = r'runtime_result/evosuit160_sf110_result_html_with_project.csv'
    metric_file_path = r'sf110_csvs_without_test_e3/102_squirrel-sql.csv'

    class_file = pd.read_csv(class_file_path, index_col=False)
    metric_file = pd.read_csv(metric_file_path, index_col=False)

    df1 = class_file.loc[class_file.Project == '102_squirrel-sql']
    df1_calsses = df1['Class']
    metric_file_classes = metric_file['Class']

    differ = set(df1_calsses).difference(set(metric_file_classes))
    print(len(differ))
    print(differ)


def equation7(lc=0., bc=0., tests=0.):
    coverageability = 1/2.*lc + 1/2.*bc
    alpha = [0.0100, 0.0150, 0.0075, 0.0050, 0.0025, 0.0001, 0.0500]
    testability = coverageability / ((1 + alpha[0]) ** (tests - 1))
    print('coverageability:', coverageability)
    print('testability:', round(testability, 4))

# extract_coverage()
# create_evosuite160coverage_csv()
# fix_ds050_coverage_info()
# add_coverage_ability_column()
# add_project_name()
# query()
# compare_class_file_metric_file()
equation7(lc=0.609467455621301,
          bc=0.9,
          tests=50
          )
