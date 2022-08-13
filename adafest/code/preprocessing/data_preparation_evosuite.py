"""

To extract compile time and runtime data from evo-suite dataset
"""

__version__ = '0.3.2'
__author__ = 'Morteza'


import sys
import os
import pandas as pd
import numpy as np
from scipy import stats
from sklearn import preprocessing
from imblearn.combine import SMOTEENN, SMOTETomek

import adafest

sys.path.insert(0, "D:/program files/scitools/bin/pc-win64/python")
import understand

from adafest.code.metrics.metrics_api_1 import *

from adafest.code import metrics


class TestabilityMetrics:
    """

    """

    @classmethod
    def get_class_ordinary_metrics_names(cls) -> list:
        metrics = ['AvgCyclomatic', 'AvgCyclomaticModified', 'AvgCyclomaticStrict', 'AvgEssential',
                   'MaxCyclomatic', 'MaxCyclomaticModified', 'MaxCyclomaticStrict', 'MaxEssential',
                   'SumCyclomatic', 'SumCyclomaticModified', 'SumCyclomaticStrict', 'SumEssential',
                   'CountClassBase', 'CountClassDerived', 'MaxInheritanceTree', 'PercentLackOfCohesion',
                   'CountClassCoupled', 'MaxNesting', 'CountDeclClassMethod', 'CountDeclClassVariable',
                   'CountDeclInstanceMethod', 'CountDeclInstanceVariable', 'CountDeclMethodAll', 'CountDeclMethod',
                   'AvgLineCode', 'CountLineCode', 'CountLineCodeDecl', 'CountLineCodeExe',
                   'CountSemicolon', 'CountStmt', 'CountStmtDecl', 'CountStmtExe',
                   'CountDeclMethodDefault', 'CountDeclMethodPrivate', 'CountDeclMethodProtected',
                   'CountDeclMethodPublic',
                   ]
        return metrics

    @classmethod
    def get_class_lexicon_metrics_names(cls) -> list:
        metrics = [
            'NumberOfIdentifies', 'NumberOfUniqueIdentifiers', 'NumberOfKeywords', 'NumberOfUniqueKeywords',
            'NumberOfOperatorsWithoutAssignments', 'NumberOfAssignments',
            'NumberOfUniqueOperators',
            'NumberOfDots', 'NumberOfTokens', 'NumberOfUniqueTokens',
            'NumberOfReturnAndPrintStatements',
            # 'NumberOfFunctionCalls',
            'NumberOfParameters',
        ]
        return metrics

    @classmethod
    def get_package_metrics_names(cls) -> list:
        metrics = [
            'AvgCyclomatic', 'AvgCyclomaticModified', 'AvgCyclomaticStrict', 'AvgEssential',
            'AvgLineCode', 'CountDeclClass', 'CountDeclClassMethod', 'CountDeclClassVariable',
            'CountDeclFile', 'CountDeclInstanceMethod', 'CountDeclInstanceVariable',
            'CountDeclMethodDefault', 'CountDeclMethodPrivate', 'CountDeclMethodProtected', 'CountDeclMethodPublic',
            'CountLineCode', 'CountLineCodeDecl', 'CountLineCodeExe', 'CountSemicolon',
            'CountStmt', 'CountStmtDecl', 'CountStmtExe',
            'MaxCyclomatic', 'MaxCyclomaticModified', 'MaxCyclomaticStrict', 'MaxEssential',
            'MaxNesting',
            'SumCyclomatic', 'SumCyclomaticModified', 'SumCyclomaticStrict', 'SumEssential',
        ]
        return metrics

    @classmethod
    def get_project_metrics_names(cls) -> list:
        metrics = [
            'NumberOfPackages', 'CountDeclClass', 'CountDeclMethod', 'CountDeclFile',
            'CountLineCode', 'CountLineCodeDecl', 'CountLineCodeExe',
            'CountSemicolon', 'CountStmt', 'CountStmtDecl', 'CountStmtExe',
            'MaxNesting', 'AvgLineCode', 'Knots'
        ]
        return metrics

    @classmethod
    def get_all_metrics_names(cls) -> list:
        metrics = list()
        print('project_metrics number: ', len(TestabilityMetrics.get_project_metrics_names()))
        for metric_name in TestabilityMetrics.get_project_metrics_names():
            metrics.append('Project_' + metric_name)

        print('package_metrics number: ', len(TestabilityMetrics.get_package_metrics_names()))
        for metric_name in TestabilityMetrics.get_package_metrics_names():
            metrics.append('Package_' + metric_name)

        print('class_lexicon_metrics number: ', len(TestabilityMetrics.get_class_lexicon_metrics_names()))
        for metric_name in TestabilityMetrics.get_class_lexicon_metrics_names():
            metrics.append('ClassLexicon_' + metric_name)

        print('class_ordinary_metrics number: ', len(TestabilityMetrics.get_class_ordinary_metrics_names()))
        for metric_name in TestabilityMetrics.get_class_ordinary_metrics_names():
            metrics.append('ClassOrdinary_' + metric_name)
        return metrics

    @classmethod
    def extract_all_metrics(cls, db):
        metrics = db.metric(db.metrics())
        i = 0
        for k, v in sorted(metrics.items()):
            print(k, "=", v)
            i += 1
        print('number of metrics', i)

    @classmethod
    def extract_class(cls, db):
        classes_list = UnderstandUtility.get_project_classes_longnames_java(db=db)
        print('-' * 75)
        print('@understand', len(set(classes_list)), set(classes_list))
        return classes_list

    # Deprecated
    @classmethod
    def compute_java_class_metrics(cls, db, class_name):
        """
        Strategy #1: Lookup understand db for given class_name
        Bug: This method do not work correctly for some projects, e.g., '17_inspirento', '82_gaj',

        :param db:
        :param class_name:
        :return:
        """
        print('Enter compute java class metric for class: ', class_name)
        entity = db.lookup(class_name + '$', 'Class')
        print('number of founded entities in lookup process: ', len(entity), entity)
        # print(entity[0].contents())
        if len(entity) != 1:
            return None
        entity = entity[0]
        metrics = entity.metric(entity.metrics())
        # print('number of metrics:', len(metrics), metrics)
        # for i, metric in enumerate(metrics.keys()):
        #     print(i + 1, ': ', metric, metrics[metric])
        # print('$%$', metrics['AvgCyclomatic'])

        # testability_metrics_dict = dict()
        # dict.update({'AvgCyclomatic': metrics['AvgCyclomatic']})

        return metrics

    @classmethod
    def compute_java_class_metrics2(cls, db=None, entity=None):
        """
        Strategy #2: Take a list of all classes and search for target class

        Which strategy is used for our final setting? I do not know!

        :param db:
        :param entity:
        :return:
        """

        metrics = entity.metric(entity.metrics())
        # print('number of metrics:', len(metrics), metrics)
        # for i, metric in enumerate(metrics.keys()):
        #     print(i + 1, ': ', metric, metrics[metric])
        # print('$%$', metrics['AvgCyclomatic'])

        # testability_metrics_dict = dict()
        # dict.update({'AvgCyclomatic': metrics['AvgCyclomatic']})
        return metrics

    @classmethod
    def compute_java_class_metrics_lexicon(cls, db=None, entity=None):
        """

        :param db:
        :param entity:
        :return:
        """
        class_lexicon_metrics_dict = dict()

        # for ib in entity.ib():
        #     print('entity ib', ib)

        # Compute lexicons
        tokens_list = list()
        identifiers_list = list()
        keywords_list = list()
        operators_list = list()
        return_and_print_count = 0
        dots_count = 0
        error_count = 0
        try:
            # print('ec', entity.parent().id())
            # source_file_entity = db.ent_from_id(entity.parent().id())

            # print('file', type(source_file_entity), source_file_entity.longname())
            for lexeme in entity.lexer(show_inactive=False):
                # print(lexeme.text(), ': ', lexeme.token())
                tokens_list.append(lexeme.text())
                if lexeme.token() == 'Identifier':
                    identifiers_list.append(lexeme.text())
                if lexeme.token() == 'Keyword':
                    keywords_list.append(lexeme.text())
                if lexeme.token() == 'Operator':
                    operators_list.append(lexeme.text())
                if lexeme.text() == 'return' or lexeme.text() == 'printf':
                    return_and_print_count += 1
                if lexeme.text() == '.':
                    dots_count += 1
        except:
            print('Error!!!', error_count)
            error_count += 1

        number_of_assignments = operators_list.count('=')
        number_of_operators_without_assignments = len(operators_list) - number_of_assignments
        number_of_unique_operators = len(set(list(filter('='.__ne__, operators_list))))

        class_lexicon_metrics_dict.update({'NumberOfTokens': len(tokens_list)})
        class_lexicon_metrics_dict.update({'NumberOfUniqueTokens': len(set(tokens_list))})

        class_lexicon_metrics_dict.update({'NumberOfIdentifies': len(identifiers_list)})
        class_lexicon_metrics_dict.update({'NumberOfUniqueIdentifiers': len(set(identifiers_list))})

        class_lexicon_metrics_dict.update({'NumberOfKeywords': len(keywords_list)})
        class_lexicon_metrics_dict.update({'NumberOfUniqueKeywords': len(set(keywords_list))})

        class_lexicon_metrics_dict.update(
            {'NumberOfOperatorsWithoutAssignments': number_of_operators_without_assignments})
        class_lexicon_metrics_dict.update({'NumberOfAssignments': number_of_assignments})
        class_lexicon_metrics_dict.update({'NumberOfUniqueOperators': number_of_unique_operators})

        class_lexicon_metrics_dict.update({'NumberOfReturnAndPrintStatements': return_and_print_count})
        class_lexicon_metrics_dict.update({'NumberOfDots': dots_count})

        # Compute NumberOfParameters
        number_of_parameters = 0
        method_list = UnderstandUtility.get_method_of_class_java(db=db, class_name=entity.longname())
        # print('method list', len(method_list))
        for method in method_list:
            # if method.library() != "Standard":
            # print('method params', method.longname(), '-->', method.parameters())
            params = method.parameters().split(',')
            if len(params) == 1:
                if params[0] == ' ' or params[0] == '' or params[0] is None:
                    number_of_parameters += 0
                else:
                    number_of_parameters += 1
            else:
                number_of_parameters += len(params)

        # print('number of parameters', number_of_parameters)

        class_lexicon_metrics_dict.update({'NumberOfParameters': number_of_parameters})
        # class_lexicon_metrics_dict.update({'NumberOfFunctionCalls': number_of_function_calls})

        # print('Lexicon metrics', class_lexicon_metrics_dict)

        return class_lexicon_metrics_dict

    @classmethod
    def compute_java_package_metrics(cls, db, class_name):

        # print('ib', entity.ib())
        # package_name = ''
        # Find package: strategy 1
        # for ib in entity.ib():
        #     if ib.find('Package:') != -1:
        #         sp = ib.split(':')
        # print('entity ib', sp[1][1:-1])
        # package_name = sp[1][1:-1]

        # Find package: strategy 2: Dominated strategy
        class_name_list = class_name.split('.')[:-1]
        package_name = '.'.join(class_name_list)
        # print('package_name string', package_name)
        package_list = db.lookup(package_name + '$', 'Package')
        if package_list is None:
            return None
        if len(package_list) == 0:  # if len != 1 return None!
            return None
        package = package_list[0]
        # print('kind:', package.kind())

        # Print info
        # print('package metrics')
        metrics = package.metric(package.metrics())
        # print('number of metrics:', len(metrics), metrics)
        # for i, metric in enumerate(metrics.keys()):
        #     print(i + 1, ': ', metric, metrics[metric])

        # print('class metrics')
        # metrics2 = entity.metric(entity.metrics())
        # print('number of metrics:', len(metrics), metrics2)
        # for i, metric2 in enumerate(metrics.keys()):
        #     print(i + 1, ': ', metric2, metrics[metric2])

        #
        # print(package.refs('Definein'))
        # for defin in package.refs('Definein'):
        #     print('kind', defin.ent().kind())
        # print(defin, '-->', defin.ent().ents('Java Define', 'Class'))
        # metrics = entity.metric(defin.ent().metrics())
        # print('number of metrics in file:', len(metrics), metrics)
        # for i, metric in enumerate(metrics.keys()):
        #     print(i + 1, ': ', metric, metrics[metric])

        return metrics

    @classmethod
    def compute_java_project_metrics(cls, db):
        project_metrics = db.metric(db.metrics())
        # print('number of metrics:', len(project_metrics),  project_metrics)
        # for i, metric in enumerate( project_metrics.keys()):
        #     print(i + 1, ': ',  metric,  project_metrics[metric])

        # print(project_metrics)
        packages = db.ents('Java Package')
        # print('number of packages', len(packages))
        project_metrics.update({'NumberOfPackages': len(packages)})

        return project_metrics

    @classmethod
    def get_entity_kind(cls, db, class_name):
        entity = db.lookup(class_name + '$', 'Type')
        return entity[0].kindname()


# ------------------------------------------------------------------------
class PreProcess:
    """

    """

    @classmethod
    def create_understand_database_from_project(cls, root_path=None):
        # First path
        # root_path = 'E:/LSSDS/EvoSuite/SF110-20130704-src/SF110-20130704-src/'

        # Second path, after eliminating all test class form SF110
        root_path = 'sf110_without_test/'  # Place fo both project sources and understand databases

        # 'create -db C:\Users\NOLIMIT\Desktop\sbta -languages c# add C:\Users\NOLIMIT\Desktop\sbta analyze -all'
        # {0}: understand_db_directory, {1}: understand_db_name, {2}: project_root_directory
        cmd = 'und create -db {0}{1}.udb -languages java add {2} analyze -all'
        # projects = [x[0] for x in os.walk(root_path)]
        projects = [name for name in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, name))]
        for project_ in projects:
            command_ = cmd.format(root_path, project_, root_path + project_)
            print('executing command {0}'.format(command_))
            # returned_value_in_byte = subprocess.check_output(command_, shell=True)
            os.system('cmd /c "{0}"'.format(command_))
            # os.system('cmd / k "{0}"'.format(command_))

    @classmethod
    def extract_metrics_and_coverage_all(cls, udbs_path, dataset_path):
        # for filename in os.listdir(udbs_path):
        #     print('file: ', filename)
        #     file_name_, file_extension_ = os.path.splitext(os.path.basename(udbs_path + filename))
        #     if file_extension_ == '.udb':
        #         pass
        # print('processing # :', udbs_path + filename[:-4])
        # db = understand.open(udbs_path + filename)
        # PreProcess.extract_metrics_and_coverage(dataset=dataset_path, database=db, project_name=filename[:-4])
        # print('-' * 75)
        # for root, dirs, files in os.walk(udbs_path):
        #     for filename in files:
        #         print(filename)
        files = [f for f in os.listdir(udbs_path) if os.path.isfile(os.path.join(udbs_path, f))]
        for f in files:
            print('processing file {0}:'.format(f))
            db = understand.open(os.path.join(udbs_path, f))
            PreProcess.extract_metrics_and_coverage(dataset=dataset_path, database=db, project_name=f[:-4])
            print('processing file {0} was finished'.format(f))

    @classmethod
    def extract_metrics_and_coverage(cls, dataset, database, project_name):
        with open(dataset, mode='r', encoding='utf8') as f:
            data_lines = f.readlines()

        # Just for test
        # for line in data_lines[:10]:
        #     items = line.split(' ')
        # print('Project ID', items[2], 'Target class', items[3],  ' -- Branch coverage', items[7])
        # print(len(items))
        # print(items)
        # quit()
        classes_coverage_list = list()
        classes_list = list()

        line_coverage_dict = dict()  # column 6,
        branch_coverage_dict = dict()  # column 7, begin from Zero
        weak_mutation_score_dict = dict()  # column 9,

        for line in data_lines:
            items = line.split(' ')
            if items[2][1:-1] == project_name:  # '102_squirrel-sql':
                long_class_name = items[3][1:-1]
                classes_coverage_list.append((items[3][1:-1], items[7]))
                classes_list.append(items[3][1:-1])

                if long_class_name in line_coverage_dict.keys():
                    line_coverage_dict[long_class_name].append(int(items[6]))
                else:
                    line_coverage_dict.update({long_class_name: [int(items[6])]})

                if long_class_name in branch_coverage_dict.keys():
                    branch_coverage_dict[long_class_name].append(float(items[7]))
                else:
                    branch_coverage_dict.update({long_class_name: [float(items[7])]})

                if long_class_name in weak_mutation_score_dict.keys():
                    weak_mutation_score_dict[long_class_name].append(float(items[9]))
                else:
                    weak_mutation_score_dict.update({long_class_name: [float(items[9])]})

        # c = Counter(classes_list)
        # print('Counter', c)
        # print('line_coverage', line_coverage_dict)
        # print('branch_coverage', branch_coverage_dict)
        # print('weak_mutation_score', weak_mutation_score_dict)

        f = open('sf110_csvs_without_test/' + project_name + '.csv', mode='w', encoding='utf8')
        f.write('class,')
        metrics_name_list = TestabilityMetrics.get_all_metrics_names()
        for metric_name in metrics_name_list:
            f.write(metric_name + ',')
        f.write('Label_LineCoverage, Label_BranchCoverage, Label_MutationScore\n')

        row = 0
        enumerator_number = 0

        # Get project metrics
        print('@ getting project metrics')
        project_metrics_dict = TestabilityMetrics.compute_java_project_metrics(db=database)
        if project_metrics_dict is None:
            print('No project metrics for project {} was found!'.format(project_name))
            return

        for item in line_coverage_dict.keys():
            if TestabilityMetrics.get_entity_kind(db=database, class_name=item).find('Enum') != -1:
                enumerator_number += 1
                print('The enum entity with name {} was found!'.format(item))
                continue

            # Find relevant class entity
            entities = UnderstandUtility.get_project_types_java(db=database)
            entity = None
            for entity_ in entities:
                if entity_.longname() == item:
                    entity = entity_
                    break
            if entity is None:
                print('No class entity with name {} was found!'.format(item))
                continue

            # Compute the average of coverage
            sum_ = 0
            for result in line_coverage_dict[item]:
                sum_ += result
            avg = sum_ / len(line_coverage_dict[item])
            line_coverage_dict[item] = round(avg)

            sum_ = 0
            for result in branch_coverage_dict[item]:
                sum_ += result
            avg = sum_ / len(branch_coverage_dict[item])
            branch_coverage_dict[item] = avg

            sum_ = 0
            for result in weak_mutation_score_dict[item]:
                sum_ += result
            avg = sum_ / len(weak_mutation_score_dict)
            weak_mutation_score_dict[item] = avg

            # Get metrics_dicts (name, value)
            # Project metrics pull-up

            # package
            print('@ getting package metrics')
            package_metrics_dict = TestabilityMetrics.compute_java_package_metrics(db=database, class_name=item)
            if package_metrics_dict is None:
                print('No package metric for item {} was found'.format(item))
                continue

            # class_lexicon
            print('@ getting lexicon metrics')
            class_lexicon_metrics_dict = TestabilityMetrics.compute_java_class_metrics_lexicon(db=database,
                                                                                               entity=entity)
            if class_lexicon_metrics_dict is None:
                print('No class lexicon metric for item {} was found'.format(item))
                continue

            # class_ordinary
            print('@ getting class metrics')
            class_ordinary_metrics_dict = TestabilityMetrics.compute_java_class_metrics2(db=database, entity=entity)
            if class_ordinary_metrics_dict is None:
                print('No class ordinary metric for item {} was found'.format(item))
                continue

            # Write class_name: new instance
            f.write(item + ',')

            # Write project_metrics_dict
            for metric_name in TestabilityMetrics.get_project_metrics_names():
                f.write(str(project_metrics_dict[metric_name]) + ',')

            # Write package_metrics_dict
            for metric_name in TestabilityMetrics.get_package_metrics_names():
                f.write(str(package_metrics_dict[metric_name]) + ',')

            # Write class_lexicon_metrics_dict
            for metric_name in TestabilityMetrics.get_class_lexicon_metrics_names():
                f.write(str(class_lexicon_metrics_dict[metric_name]) + ',')

            # Write class_ordinary_metrics_dict
            for metric_name in TestabilityMetrics.get_class_ordinary_metrics_names():
                f.write(str(class_ordinary_metrics_dict[metric_name]) + ',')

            # Write instance labels
            f.write(str(line_coverage_dict[item]) + ',')
            f.write(str(branch_coverage_dict[item]) + ',')
            f.write(str(weak_mutation_score_dict[item]) + '\n')

            row += 1
            print('writing row: {0}, number of enumeration: {1}'.format(row, enumerator_number))

        f.close()

        # print('line_coverage', line_coverage_dict)
        # print('branch_coverage', branch_coverage_dict)
        # print('weak_mutation_score', weak_mutation_score_dict)

        # return classes_coverage_list, set(classes_list)

    @classmethod
    def create_complete_dataset(cls, ):
        """
        This method merge all separate csv files which belongs to each project
        into one csv file for using with machine learning classifiers
        :return:
        """
        # csvs_path = 'csvs/' # path csv data for dataset version 0
        csvs_path = 'sf110_csvs_without_test/'  # path of csv data for dataset version 1

        # csv_dataset_path = 'es_complete_dataset_all_1_0_6_withtest_36col.csv'  # path of dataset 0
        csv_dataset_path = 'es_complete_dataset_all_1_0_6_without_test_93col.csv'  # path of dataset 1

        data_file = open(csv_dataset_path, mode='w', encoding='utf8')
        data_file.write('Class,')
        metrics_name = TestabilityMetrics.get_all_metrics_names()
        for metric_name in metrics_name:
            data_file.write(metric_name + ',')
        data_file.write('Label_LineCoverage,Label_BranchCoverage,Label_MutationScore\n')

        for filename in os.listdir(csvs_path):
            with open(csvs_path + filename, mode='r', encoding='utf8') as f:
                lines = f.readlines()
            data_file.writelines(lines[1:])
        data_file.close()

    @classmethod
    def intersection(cls, list1, list2):
        return list(set(list1) & set(list2))

    @classmethod
    def add_one_metric_to_csv(cls, udbs_path, current_csv_dir_path):
        files = [f for f in os.listdir(udbs_path) if os.path.isfile(os.path.join(udbs_path, f))]
        for f in files:
            print('processing file {0}:'.format(f))
            db = understand.open(os.path.join(udbs_path, f))
            df = pd.read_csv(os.path.join(current_csv_dir_path, f[:-4]+'csv'),
                             delimiter=',',
                             index_col=False,
                             # usecols=[0,2],
                            )
            df.columns = [column.replace(' ', '') for column in df.columns]

            all_metrics_list = []
            current_metric_list = []
            # print('f', f)
            # quit()

            for class_name in df['class']:
                # Find relevant class entity
                print('enter class_name: '.format(class_name))
                entities = UnderstandUtility.get_project_types_java(db=db)
                entity = None
                for entity_ in entities:
                    if entity_.longname() == class_name:
                        entity = entity_
                        break
                if entity is None:
                    print('No class entity with name {} was found!'.format(class_name))
                    continue
                cm = ClassMetric(db=db, project_name=f[:-4], package_name=None, class_name=class_name)
                for metric_name in adafest.code.metrics.metrics_names.class_cyclomatic_complexity:
                    current_metric_list.append(cm.compute_metric(metric_name=metric_name))
                all_metrics_list.append(current_metric_list)

            for i, metric_name in enumerate(adafest.code.metrics.metrics_names.class_cyclomatic_complexity):
                df[metric_name] = all_metrics_list[i]

            new_csv_dir_path = r''
            df.to_csv(os.path.join(new_csv_dir_path, f), index=False)
            print('processing file {0} was finished'.format(f))


    # Data cleaning
    @classmethod
    def quantile_based_discretization(cls, path):
        """

        :return:
        """
        data_frame = pd.read_csv(path,
                                 # delimiter=',',
                                 # index_col=False,
                                 # usecols=[0,2]
                                 )
        # data_frame.columns = [column.replace(' ', '_') for column in data_frame.columns]
        # print(data_frame)
        # quit()
        # Add Testability column
        labels = ['VeryLow', 'Low', 'Mean', 'High', 'VeryHigh']
        data_frame['Testability'] = pd.cut(data_frame.loc[:, ['Label_BranchCoverage']].T.squeeze(),
                                           bins=5,
                                           labels=labels
                                           )
        # print(pd.cut(data.loc[:, ['_branch_coverage']].T.squeeze(),
        #              bins=5,
        #              labels=['VeryLow', 'Low', 'Mean', 'High', 'VeryHigh']
        #              ).value_counts())
        print(data_frame)

        # Remove extra columns
        data_frame_dropped = data_frame.drop(['Label_LineCoverage', 'Label_BranchCoverage', 'Label_MutationScore'], axis=1)
        # Save new dataset
        print(data_frame_dropped)
        data_frame_dropped.to_csv(r'es_complete_dataset_all_1_0_6_without_test_93col_discretize_91col_15417.csv',
                                  index=False)
        # data_frame.to_html(r'es_complete_dataset_all_1_0_6_without_test_93col_discretize_91col.html)


    @classmethod
    def mitigate_imbalanced(cls, path):
        # pd.set_option("display.max_rows", None, "display.max_columns", None)
        pd.options.display.max_colwidth = 1000
        data = pd.read_csv(path, delimiter=',',
                           index_col=False,
                           # usecols=[0,2],
                           )
        print(type(data))
        data.columns = [column.replace(' ', '_') for column in data.columns]

        # df = pd.DataFrame(data, columns=['class', "_Project_CountDeclClass"])
        # df = pd.DataFrame(data=data,)
        # print(df)
        # for i, j in df.iterrows():
        #     print(i, j)
        #     print()

        # print(df.isna())
        # print(df.query("_branch_coverage==1.0"))
        # print(data.columns)
        # print(data.loc[1:50, "_branch_coverage")
        # data.filter()
        # data1 = data["_branch_coverage"] == 1.
        # data1 = data1[0:50, "_branch_coverage"]
        data1 = data.loc[(data['Label_BranchCoverage'] >= 1.)
                    # & (data['ClassOrdinary_CountLineCode'] == 10e6)
                    # & (data['class'].str.contains('weka'))
                    #& (data.ClassOrdinary_MaxNesting == data.ClassOrdinary_MaxNesting.max())
                    & (data.ClassOrdinary_MaxCyclomatic <= 1)
                    # & (data['ClassOrdinary_CountDeclMethodAll'] <= 5)
                    #  & (data.ClassOrdinary_CountDeclMethod <= 0)
                    # & (data.ClassOrdinary_MaxNesting == 0)
                         ]
        # data1 = data1.filter(like='UnifyCase', axis=0)
        # data1 = data1[data1.ClassOrdinary_MaxNesting == data1.ClassOrdinary_MaxNesting.max()]
        # data1 = data1[data1.ClassOrdinary_AvgCyclomatic == data1.ClassOrdinary_AvgCyclomatic.max()]
        # data1 = data1[data1.ClassOrdinary_SumCyclomatic == data1.ClassOrdinary_SumCyclomatic.max()]
        # data1 = data1[data1.ClassOrdinary_CountLineCode == data1.ClassOrdinary_CountLineCode.max()]
        print('data1', data1.shape)

        data2 = pd.concat([data, data1]).drop_duplicates(keep=False)
        print('data2', data2.shape)

        data3 = data2.loc[
            (data['Label_BranchCoverage'] < 1.)
            & (data.ClassOrdinary_CountDeclMethodAll >= 0)
        ]

        print(data2.shape)

        data3 = data2.loc[:, ['Class',
                              'ClassOrdinary_CountLineCode',
                              'ClassOrdinary_AvgCyclomatic',
                              'ClassOrdinary_SumCyclomatic',
                              'ClassOrdinary_MaxNesting',
                              'Label_BranchCoverage']]
        print(data2.shape)

        # col = data['ClassOrdinary_CountLineCode']
        # print('max col is in row {0} with value {1} and name {3}'.format(col.idxmax(), col.max(), col[col.idxmax(), :]))
        # print(data.max())
        # print(data[data.ClassOrdinary_CountLineCode ==
        #            data.ClassOrdinary_CountLineCode.max()][['Class', 'Label_LineCoverage']])



        data2.to_csv('temp_data_pandas_filter.csv', header=True, index=True, index_label='index')

    @classmethod
    def removing_outliers(cls, path):
        # pd.set_option("display.max_rows", None, "display.max_columns", None)
        pd.options.display.max_colwidth = 1000
        df = pd.read_csv(path,
                           delimiter=',',
                           index_col=False,
                           # usecols=[0,2]
                           )
        print(type(df))
        df.columns = [column.replace(' ', '_') for column in df.columns]

        df2 = df.drop(columns=['Testability', 'Class'])
        z_scores = stats.zscore(df2)
        # print(z_scores)
        abs_z_scores = np.abs(z_scores)
        filtered_entries = (abs_z_scores < 3).all(axis=1)
        new_df = df[filtered_entries]  # we can use df or df2
        print(new_df)
        new_df.to_csv('es_complete_dataset_all_1_0_6_without_test_93col_discretize_91col_15417_outlier_removed.csv', index=False)

    @classmethod
    def normalize_features(cls, path):
        """
        https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html#sphx-glr-auto-examples-preprocessing-plot-all-scaling-py
        :param path:
        :return:
        """
        # pd.set_option("display.max_rows", None, "display.max_columns", None)
        pd.options.display.max_colwidth = 1000
        df = pd.read_csv(path,
                         delimiter=',',
                         index_col=False,
                         # usecols=[0,2]
                         )
        print(type(df))
        df.columns = [column.replace(' ', '_') for column in df.columns]

        df2 = df.drop(columns=['Testability', 'Class'])
        x = df2.values  # returns a numpy array
        min_max_scaler = preprocessing.MinMaxScaler()
        robust_scaler = preprocessing.RobustScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        new_df = pd.DataFrame(x_scaled)
        new_df['Testability'] = df['Testability']

        print(new_df)
        new_df.to_csv('es_complete_dataset_all_1_0_6_without_test_93col_discretize_91col_15417_outlier_removed_min_max_scaler.csv',
                      index=False)

    @classmethod
    def resampling(cls, path):
        # pd.set_option("display.max_rows", None, "display.max_columns", None)
        pd.options.display.max_colwidth = 1000
        df = pd.read_csv(path,
                         delimiter=',',
                         index_col=False,
                         # usecols=[0,2]
                         )
        # print(type(df))
        df.columns = [column.replace(' ', '_') for column in df.columns]

        # Strategy one: Simply duplicate the data: Dose not work on test set
        df2 = df.append(df, ignore_index=True)

        # Strategy two: Combination of over- and under-sampling
        # https://imbalanced-learn.readthedocs.io/en/stable/combine.html
        X = df.iloc[:, 2:-1]
        y = df.iloc[:, -1]

        smote_tomek = SMOTETomek(random_state=0)
        smote_enn = SMOTEENN(random_state=0, )
        # X_resampled, y_resampled = SMOTE().fit_resample(X, y)
        # X_resampled, y_resampled = ADASYN().fit_resample(X, y)

        # X_resampled, y_resampled = smote_tomek.fit_resample(X, y)
        # SMOTEENN tends to clean more noisy samples than SMOTETomek.
        X_resampled, y_resampled = smote_enn.fit_resample(X, y)

        print(sorted(Counter(y_resampled).items()))

        df2 = pd.DataFrame(X_resampled)
        df2['Testability'] = y_resampled

        print(df2)

        df2.to_csv(
            'es_complete_dataset_all_1_0_6_without_test_93col_discretize_91col_15417_outlier_removed_binary_SMOTEENN.csv',
            index=False)

    @classmethod
    def resampling_numerical_dataset(cls, path):
        pd.options.display.max_colwidth = 1000
        df = pd.read_csv(path,
                         delimiter=',',
                         index_col=False,
                         # usecols=[0,2]
                         )
        # print(type(df))
        df.columns = [column.replace(' ', '_') for column in df.columns]

        # Strategy one: Simply duplicate the data: Dose not work on test set
        df2 = df.append(df, ignore_index=True)

        # Strategy two: Combination of over- and under-sampling
        # https://imbalanced-learn.readthedocs.io/en/stable/combine.html
        df2 = df2.drop(columns=['index', 'Class', 'Label_MutationScore', 'Label_LineCoverage'])
        X = df2.iloc[:, 0:-1]
        y = df2.iloc[:, -1]

        smote_tomek = SMOTETomek(random_state=0)
        smote_enn = SMOTEENN(random_state=0, )
        # X_resampled, y_resampled = SMOTE().fit_resample(X, y)
        # X_resampled, y_resampled = ADASYN().fit_resample(X, y)

        # X_resampled, y_resampled = smote_tomek.fit_resample(X, y)
        # SMOTEENN tends to clean more noisy samples than SMOTETomek.
        X_resampled, y_resampled = smote_enn.fit_resample(X, y)

        print(sorted(Counter(y_resampled).items()))

        df2 = pd.DataFrame(X_resampled)
        df2['Testability'] = y_resampled

        print(df2)

        df2.to_csv(
            'es_complete_dataset_all_1_0_6_without_test_91col_numerical_15417_SMOTEENN.csv',
            index=False)

    @classmethod
    def add_testability_values(cls, path):
        pd.options.display.max_colwidth = 1000
        df = pd.read_csv(path,
                         delimiter=',',
                         index_col=False,
                         # usecols=[0,2]
                         )
        # print(type(df))
        df.columns = [column.replace(' ', '_') for column in df.columns]
        # testability_labels = {'VeryLow': 10, 'Low': 30, 'Mean': 50, 'High': 70, 'VeryHigh': 90}
        # testability_labels = {'VeryLow': 0, 'Low': 0, 'Mean': 0, 'High': 1, 'VeryHigh': 1}
        testability_labels = {'VeryLow': 'NonTestable', 'Low': 'NonTestable', 'Mean': 'NonTestable', 'High': 'Testable', 'VeryHigh': 'Testable'}
        df.Testability = [testability_labels[item] for item in df.Testability]

        print(df)
        df.to_csv(r'es_complete_dataset_all_1_0_6_without_test_93col_discretize_91col_15417_outlier_removed_binary.csv',
                  index=False)

