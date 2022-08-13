"""

To extract compile time and runtime data from evo-suite dataset
"""
import sys
import os
import pandas as pd
import numpy as np
from scipy import stats
from sklearn import preprocessing
from imblearn.combine import SMOTEENN, SMOTETomek
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor

from adafest.code.metrics import metrics_names

sys.path.insert(0, "D:/program files/scitools/bin/pc-win64/python")
import understand

from adafest.code.metrics.metrics_api_1 import *

from adafest.code import metrics
from adafest.code.metrics.metrics_jcodeodor import JCodeOdorMetric
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.decomposition import PCA

class TestabilityMetrics:
    """

    """

    @classmethod
    def get_class_ordinary_metrics_names(cls) -> list:
        return adafest.code.metrics.metrics_names.class_ordinary_metrics_names

    @classmethod
    def get_class_lexicon_metrics_names(cls) -> list:
        return adafest.code.metrics.metrics_names.class_lexicon_metrics_names

    @classmethod
    def get_package_metrics_names(cls) -> list:
        return adafest.code.metrics.metrics_names.package_metrics_names

    @classmethod
    def get_project_metrics_names(cls) -> list:
        return adafest.code.metrics.metrics_names.project_metrics_names

    @classmethod
    def get_all_metrics_names(cls) -> list:
        metrics = list()
        print('project_metrics number: ', len(TestabilityMetrics.get_project_metrics_names()))
        for metric_name in TestabilityMetrics.get_project_metrics_names():
            metrics.append('PJ_' + metric_name)

        print('package_metrics number: ', len(TestabilityMetrics.get_package_metrics_names()))
        for metric_name in TestabilityMetrics.get_package_metrics_names():
            metrics.append('PK_' + metric_name)

        # Sooti
        print('class_lexicon_metrics number: ', len(TestabilityMetrics.get_class_lexicon_metrics_names()))
        for metric_name in TestabilityMetrics.get_class_lexicon_metrics_names():
            metrics.append('CSLEX_' + metric_name)

        print('class_ordinary_metrics number: ', len(TestabilityMetrics.get_class_ordinary_metrics_names()))
        for metric_name in TestabilityMetrics.get_class_ordinary_metrics_names():
            metrics.append('CSORD_' + metric_name)

        print('All available metrics:{0}'.format(len(metrics)))
        return metrics

    @classmethod
    def get_all_primary_metrics_names(cls) -> list:
        primary_metrics_names = list()
        for metric_name in adafest.code.metrics.metrics_names.project_metrics_names_primary:
            primary_metrics_names.append('PJ_' + metric_name)
        for metric_name in adafest.code.metrics.metrics_names.package_metrics_names_primary:
            primary_metrics_names.append('PK_' + metric_name)
        for metric_name in adafest.code.metrics.metrics_names.class_ordinary_metrics_names_primary:
            primary_metrics_names.append('CSORD_' + metric_name)
        for metric_name in adafest.code.metrics.metrics_names.class_lexicon_metrics_names:
            primary_metrics_names.append('CSLEX_' + metric_name)

        return primary_metrics_names

    @classmethod
    def extract_all_understand_project_metrics(cls, db):
        metrics = db.metric(db.metrics())
        i = 0
        for k, v in sorted(metrics.items()):
            print(k, '=', v)
            i += 1
        print('Project number of metrics {0}'.format(i))

    @classmethod
    def extract_class(cls, db):
        classes_list = UnderstandUtility.get_project_classes_longnames_java(db=db)
        print('-' * 75)
        print('@understand', len(set(classes_list)), set(classes_list))
        return classes_list

    # @Deprecated
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

        # 1. Understand built-in class metrics
        class_metrics = entity.metric(entity.metrics())
        # print('number of metrics:', len(metrics), metrics)
        # for i, metric in enumerate(metrics.keys()):
        #     print(i + 1, ': ', metric, metrics[metric])
        # print('$%$', metrics['AvgCyclomatic'])

        # testability_metrics_dict = dict()
        # dict.update({'AvgCyclomatic': metrics['AvgCyclomatic']})

        # 2. Systematically created metrics
        j_code_odor_metric = JCodeOdorMetric()
        method_list = UnderstandUtility.get_method_of_class_java(db=db, class_name=entity.longname())

        # 2.1 CSCC
        class_cyclomatic_list = list()
        class_cyclomatic_namm_list = list()

        class_cyclomatic_strict_list = list()
        class_cyclomatic_strict_namm_list = list()

        class_cyclomatic_modified_list = list()
        class_cyclomatic_modified_namm_list = list()

        class_essential_list = list()
        class_essential_namm_list = list()

        for method in method_list:
            class_cyclomatic_list.append(method.metric(['Cyclomatic'])['Cyclomatic'])
            class_cyclomatic_strict_list.append(method.metric(['CyclomaticStrict'])['CyclomaticStrict'])
            class_cyclomatic_modified_list.append(method.metric(['CyclomaticModified'])['CyclomaticModified'])
            class_essential_list.append(method.metric(['Essential'])['Essential'])
            if not j_code_odor_metric.is_accesor_or_mutator(method_entity=method):
                class_cyclomatic_namm_list.append(method.metric(['Cyclomatic'])['Cyclomatic'])
                class_cyclomatic_strict_namm_list.append(method.metric(['CyclomaticStrict'])['CyclomaticStrict'])
                class_cyclomatic_modified_namm_list.append(method.metric(['CyclomaticModified'])['CyclomaticModified'])
                class_essential_namm_list.append(method.metric(['Essential'])['Essential'])
        class_cyclomatic_list = [i for i in class_cyclomatic_list if i is not None]
        class_cyclomatic_namm_list = [i for i in class_cyclomatic_namm_list if i is not None]

        class_cyclomatic_strict_list = [i for i in class_cyclomatic_strict_list if i is not None]
        class_cyclomatic_strict_namm_list = [i for i in class_cyclomatic_strict_namm_list if i is not None]

        class_cyclomatic_modified_list = [i for i in class_cyclomatic_modified_list if i is not None]
        class_cyclomatic_modified_namm_list = [i for i in class_cyclomatic_modified_namm_list if i is not None]

        class_essential_list = [i for i in class_essential_list if i is not None]
        class_essential_namm_list = [i for i in class_essential_namm_list if i is not None]

        if len(class_cyclomatic_list) == 0:
            class_cyclomatic_list.append(0)
        if len(class_cyclomatic_namm_list) == 0:
            class_cyclomatic_namm_list.append(0)
        if len(class_cyclomatic_strict_list) == 0:
            class_cyclomatic_strict_list.append(0)
        if len(class_cyclomatic_strict_namm_list) == 0:
            class_cyclomatic_strict_namm_list.append(0)
        if len(class_cyclomatic_modified_list) == 0:
            class_cyclomatic_modified_list.append(0)
        if len(class_cyclomatic_modified_list) == 0:
            class_cyclomatic_modified_list.append(0)
        if len(class_cyclomatic_modified_namm_list) == 0:
            class_cyclomatic_modified_namm_list.append(0)
        if len(class_essential_list) == 0:
            class_essential_list.append(0)
        if len(class_essential_namm_list) == 0:
            class_essential_namm_list.append(0)

        # CSCC
        # 2.1.13
        class_metrics.update({'MinCyclomatic': min(class_cyclomatic_list)})
        # 2.1.14
        class_metrics.update({'MinCyclomaticStrict': min(class_cyclomatic_strict_list)})
        # 2.1.15
        class_metrics.update({'MinCyclomaticModified': min(class_cyclomatic_modified_list)})
        # 2.1.16
        class_metrics.update({'MinEssential': min(class_essential_list)})

        # 2.1.17
        class_metrics.update({'SDCyclomatic': np.std(class_cyclomatic_list)})
        # 2.1.18
        class_metrics.update({'SDCyclomaticStrict': np.std(class_cyclomatic_strict_list)})
        # 2.1.19
        class_metrics.update({'SDCyclomaticModified': np.std(class_cyclomatic_modified_list)})
        # 2.1.20
        class_metrics.update({'SDEssential': np.std(class_essential_list)})

        # CSCCNAMM
        # 2.1.21
        class_metrics.update({'SumCyclomaticNAMM': sum(class_cyclomatic_namm_list)})
        # 2.1.22
        class_metrics.update({'SumCyclomaticStrictNAMM': sum(class_cyclomatic_strict_namm_list)})
        # 2.1.23
        class_metrics.update({'SumCyclomaticModifiedNAMM': sum(class_cyclomatic_modified_namm_list)})
        # 2.1.24
        class_metrics.update({'SumEssentialNAMM': sum(class_essential_namm_list)})

        # 2.1.25
        class_metrics.update({'MaxCyclomaticNAMM': max(class_cyclomatic_namm_list)})
        # 2.1.26
        class_metrics.update({'MaxCyclomaticStrictNAMM': max(class_cyclomatic_strict_namm_list)})
        # 2.1.27
        class_metrics.update({'MaxCyclomaticModifiedNAMM': max(class_cyclomatic_modified_namm_list)})
        # 2.1.28
        class_metrics.update({'MaxEssentialNAMM': max(class_essential_namm_list)})

        # 2.1.29
        class_metrics.update({'AvgCyclomaticNAMM': sum(class_cyclomatic_namm_list) / len(class_cyclomatic_namm_list)})
        # 2.1.30
        class_metrics.update({'AvgCyclomaticStrictNAMM': sum(class_cyclomatic_strict_namm_list) / len(
            class_cyclomatic_strict_namm_list)})
        # 2.1.31
        class_metrics.update({'AvgCyclomaticModifiedNAMM': sum(class_cyclomatic_modified_namm_list) / len(
            class_cyclomatic_modified_namm_list)})
        # 2.1.32
        class_metrics.update({'AvgEssentialNAMM': sum(class_essential_namm_list) / len(class_essential_namm_list)})

        # 2.1.33
        class_metrics.update({'MinCyclomaticNAMM': min(class_cyclomatic_namm_list)})
        # 2.1.34
        class_metrics.update({'MinCyclomaticStrictNAMM': min(class_cyclomatic_strict_namm_list)})
        # 2.1.35
        class_metrics.update({'MinCyclomaticModifiedNAMM': min(class_cyclomatic_modified_namm_list)})
        # 2.1.36
        class_metrics.update({'MinEssentialNAMM': min(class_essential_namm_list)})

        # 2.1.37
        class_metrics.update({'SDCyclomaticNAMM': np.std(class_cyclomatic_namm_list)})
        # 2.1.38
        class_metrics.update({'SDCyclomaticStrictNAMM': np.std(class_cyclomatic_strict_namm_list)})
        # 2.1.39
        class_metrics.update({'SDCyclomaticModifiedNAMM': np.std(class_cyclomatic_modified_namm_list)})
        # 2.1.40
        class_metrics.update({'SDEssentialNAMM': np.std(class_essential_namm_list)})

        # 2.2 CSNOP (10)
        #
        parameters_length_list = list()
        parameters_length_namm_list = list()
        # number_of_parameters = 0
        # print('method list', len(method_list))
        for method in method_list:
            # if method.library() != "Standard":
            # print('method params', method.longname(), '-->', method.parameters())
            params = method.parameters().split(',')
            if len(params) == 1:
                if params[0] == ' ' or params[0] == '' or params[0] is None:
                    parameters_length_list.append(0)
                else:
                    parameters_length_list.append(1)
            else:
                parameters_length_list.append(len(params))

            if not j_code_odor_metric.is_accesor_or_mutator(method_entity=method):
                if len(params) == 1:
                    if params[0] == ' ' or params[0] == '' or params[0] is None:
                        parameters_length_namm_list.append(0)
                    else:
                        parameters_length_namm_list.append(1)
                else:
                    parameters_length_namm_list.append(len(params))

        parameters_length_list = [i for i in parameters_length_list if i is not None]
        parameters_length_namm_list = [i for i in parameters_length_namm_list if i is not None]

        if len(parameters_length_list) == 0:
            parameters_length_list.append(0)
        if len(parameters_length_namm_list) == 0:
            parameters_length_namm_list.append(0)

        # print('number of parameters', number_of_parameters)
        # CSNOP
        # 2.2.1
        class_metrics.update({'SumCSNOP': sum(parameters_length_list)})
        # 2.2.2
        class_metrics.update({'MaxCSNOP': max(parameters_length_list)})
        # 2.2.3
        class_metrics.update({'MinCSNOP': min(parameters_length_list)})
        # 2.2.4
        class_metrics.update({'AvgCSNOP': sum(parameters_length_list) / len(parameters_length_list)})
        # 2.2.5
        class_metrics.update({'SDCSNOP': np.std(parameters_length_list)})

        # CSNOP_NAMM
        # 2.2.6
        class_metrics.update({'SumCSNOPNAMM': sum(parameters_length_namm_list)})
        # 2.2.7
        class_metrics.update({'MaxCSNOPNAMM': max(parameters_length_namm_list)})
        # 2.2.8
        class_metrics.update({'MinCSNOPNAMM': min(parameters_length_namm_list)})
        # 2.2.9
        class_metrics.update({'AvgCSNOPNAMM': sum(parameters_length_namm_list) / len(parameters_length_namm_list)})
        # 2.2.10
        class_metrics.update({'SDCSNOPNAMM': np.std(parameters_length_namm_list)})

        # 2.3 SCLOC (30)
        #
        line_of_code_list = list()
        line_of_code_namm_list = list()

        line_of_code_decl_list = list()
        line_of_code_decl_namm_list = list()

        line_of_code_exe_list = list()
        line_of_code_exe_namm_list = list()
        for method in method_list:
            line_of_code_list.append(method.metric(['CountLineCode'])['CountLineCode'])
            line_of_code_decl_list.append(method.metric(['CountLineCodeDecl'])['CountLineCodeDecl'])
            line_of_code_exe_list.append(method.metric(['CountLineCodeExe'])['CountLineCodeExe'])
            if not j_code_odor_metric.is_accesor_or_mutator(method_entity=method):
                line_of_code_namm_list.append(method.metric(['CountLineCode'])['CountLineCode'])
                line_of_code_decl_namm_list.append(method.metric(['CountLineCodeDecl'])['CountLineCodeDecl'])
                # print('!@#', method.metric(['CountLineCodeDecl'])['CountLineCodeDecl'])
                # quit()
                line_of_code_exe_namm_list.append(method.metric(['CountLineCodeExe'])['CountLineCodeExe'])
        line_of_code_list = [i for i in line_of_code_list if i is not None]
        line_of_code_namm_list = [i for i in line_of_code_namm_list if i is not None]
        line_of_code_decl_list = [i for i in line_of_code_decl_list if i is not None]
        line_of_code_decl_namm_list = [i for i in line_of_code_decl_namm_list if i is not None]
        line_of_code_exe_list = [i for i in line_of_code_exe_list if i is not None]
        line_of_code_exe_namm_list = [i for i in line_of_code_exe_namm_list if i is not None]

        # print('!@#', line_of_code_decl_namm_list)
        # quit()
        if len(line_of_code_list) == 0:
            line_of_code_list.append(0)
        if len(line_of_code_decl_list) == 0:
            line_of_code_decl_list.append(0)
        if len(line_of_code_exe_list) == 0:
            line_of_code_exe_list.append(0)

        if len(line_of_code_namm_list) == 0:
            line_of_code_namm_list.append(0)
        if len(line_of_code_exe_namm_list) == 0:
            line_of_code_exe_namm_list.append(0)
        if len(line_of_code_decl_namm_list) == 0:
            line_of_code_decl_namm_list.append(0)

        # CSLOC_All

        # 2.3.5
        class_metrics.update({'AvgLineCodeDecl': sum(line_of_code_decl_list) / len(line_of_code_decl_list)})
        # 2.3.6
        class_metrics.update({'AvgLineCodeExe': sum(line_of_code_exe_list) / len(line_of_code_exe_list)})

        # 2.3.7
        class_metrics.update({'MaxLineCode': max(line_of_code_list)})
        # 2.3.8
        class_metrics.update({'MaxLineCodeDecl': max(line_of_code_decl_list)})
        # 2.3.9

        class_metrics.update({'MaxLineCodeExe': max(line_of_code_exe_list)})

        # 2.3.10
        class_metrics.update({'MinLineCode': min(line_of_code_list)})
        # 2.3.11
        class_metrics.update({'MinLineCodeDecl': min(line_of_code_decl_list)})
        # 2.3.12
        class_metrics.update({'MinLineCodeExe': min(line_of_code_exe_list)})

        # 2.3.13
        class_metrics.update({'SDLineCode': np.std(line_of_code_list)})
        # 2.3.14
        class_metrics.update({'SDLineCodeDecl': np.std(line_of_code_decl_list)})
        # 2.3.15
        class_metrics.update({'SDLineCodeExe': np.std(line_of_code_exe_list)})

        # CSLOC_NAMM
        # 2.3.16
        class_metrics.update({'CountLineCodeNAMM': sum(line_of_code_namm_list)})
        # 2.3.17
        class_metrics.update({'CountLineCodeDeclNAMM': sum(line_of_code_decl_namm_list)})

        # print('!@#', sum(line_of_code_decl_namm_list))
        # quit()

        # 2.3.18
        class_metrics.update({'CountLineCodeExeNAMM': sum(line_of_code_exe_namm_list)})

        # 2.3.19
        class_metrics.update({'AvgLineCodeNAMM': sum(line_of_code_namm_list) / len(line_of_code_namm_list)})
        # 2.3.20
        class_metrics.update(
            {'AvgLineCodeDeclNAMM': sum(line_of_code_decl_namm_list) / len(line_of_code_decl_namm_list)})
        # 2.3.21
        class_metrics.update({'AvgLineCodeExeNAMM': sum(line_of_code_exe_namm_list) / len(line_of_code_exe_namm_list)})

        # 2.3.22
        class_metrics.update({'MaxLineCodeNAMM': max(line_of_code_namm_list)})
        # 2.3.23
        class_metrics.update({'MaxLineCodeDeclNAMM': max(line_of_code_decl_namm_list)})
        # 2.3.24
        class_metrics.update({'MaxLineCodeExeNAMM': max(line_of_code_exe_namm_list)})

        # 2.3.25
        class_metrics.update({'MinLineCodeNAMM': min(line_of_code_namm_list)})
        # 2.3.26
        class_metrics.update({'MinLineCodeDeclNAMM': min(line_of_code_decl_namm_list)})
        # 2.3.27
        class_metrics.update({'MinLineCodeExeNAMM': min(line_of_code_exe_namm_list)})

        # 2.3.28
        class_metrics.update({'SDLineCodeNAMM': np.std(line_of_code_namm_list)})
        # 2.3.29
        class_metrics.update({'SDLineCodeDeclNAMM': np.std(line_of_code_decl_namm_list)})
        # print('!@#', np.std(line_of_code_decl_namm_list))
        # quit()
        # 2.3.30
        class_metrics.update({'SDLineCodeExeNAMM': np.std(line_of_code_exe_namm_list)})

        # ----------------------------------------------------------------
        # 2.4 CSNOST (3-->30)
        # To be completed in future work
        number_of_stmt_list = list()
        number_of_stmt_namm_list = list()

        number_of_stmt_decl_list = list()
        number_of_stmt_decl_namm_list = list()

        number_of_stmt_exe_list = list()
        number_of_stmt_exe_namm_list = list()

        for method in method_list:
            number_of_stmt_list.append(method.metric(['CountStmt'])['CountStmt'])
            number_of_stmt_decl_list.append(method.metric(['CountStmtDecl'])['CountStmtDecl'])
            number_of_stmt_exe_list.append(method.metric(['CountStmtExe'])['CountStmtExe'])
            if not j_code_odor_metric.is_accesor_or_mutator(method_entity=method):
                number_of_stmt_namm_list.append(method.metric(['CountStmt'])['CountStmt'])
                number_of_stmt_decl_namm_list.append(method.metric(['CountStmtDecl'])['CountStmtDecl'])
                number_of_stmt_exe_namm_list.append(method.metric(['CountStmtExe'])['CountStmtExe'])

        number_of_stmt_list = [i for i in number_of_stmt_list if i is not None]
        number_of_stmt_namm_list = [i for i in number_of_stmt_namm_list if i is not None]

        number_of_stmt_decl_list = [i for i in number_of_stmt_decl_list if i is not None]
        number_of_stmt_decl_namm_list = [i for i in number_of_stmt_decl_namm_list if i is not None]

        number_of_stmt_exe_list = [i for i in number_of_stmt_exe_list if i is not None]
        number_of_stmt_exe_namm_list = [i for i in number_of_stmt_exe_namm_list if i is not None]

        if len(number_of_stmt_list) == 0:
            number_of_stmt_list.append(0)
        if len(number_of_stmt_namm_list) == 0:
            number_of_stmt_namm_list.append(0)
        if len(number_of_stmt_decl_list) == 0:
            number_of_stmt_decl_list.append(0)
        if len(number_of_stmt_decl_namm_list) == 0:
            number_of_stmt_decl_namm_list.append(0)
        if len(number_of_stmt_exe_list) == 0:
            number_of_stmt_exe_list.append(0)
        if len(number_of_stmt_exe_namm_list) == 0:
            number_of_stmt_exe_namm_list.append(0)

        # CSNOST_All
        # 2.4.4
        class_metrics.update({'AvgStmt': sum(number_of_stmt_list) / len(number_of_stmt_list)})
        # 2.4.5
        class_metrics.update({'AvgStmtDecl': sum(number_of_stmt_decl_list) / len(number_of_stmt_decl_list)})
        # 2.4.6
        class_metrics.update({'AvgStmtExe': sum(number_of_stmt_exe_list) / len(number_of_stmt_exe_list)})

        # 2.4.7
        class_metrics.update({'MaxStmt': max(number_of_stmt_list)})
        # 2.4.8
        class_metrics.update({'MaxStmtDecl': max(number_of_stmt_decl_list)})
        # 2.4.9
        class_metrics.update({'MaxStmtExe': max(number_of_stmt_exe_list)})

        # 2.4.10
        class_metrics.update({'MinStmt': min(number_of_stmt_list)})
        # 2.4.11
        class_metrics.update({'MinStmtDecl': min(number_of_stmt_decl_list)})
        # 2.4.12
        class_metrics.update({'MinStmtExe': min(number_of_stmt_exe_list)})

        # 2.4.13
        class_metrics.update({'SDStmt': np.std(number_of_stmt_list)})
        # 2.4.14
        class_metrics.update({'SDStmtDecl': np.std(number_of_stmt_decl_list)})
        # 2.4.15
        class_metrics.update({'SDStmtExe': np.std(number_of_stmt_exe_list)})

        # CSNOST_NAMM
        # 2.4.16
        class_metrics.update({'CountStmtNAMM': sum(number_of_stmt_namm_list)})
        # 2.4.17
        class_metrics.update({'CountStmtDeclNAMM': sum(number_of_stmt_decl_namm_list)})
        # 2.4.18
        class_metrics.update({'CountStmtExeNAMM': sum(number_of_stmt_exe_namm_list)})

        # 2.4.19
        class_metrics.update({'AvgStmtNAMM': sum(number_of_stmt_namm_list) / len(number_of_stmt_namm_list)})
        # 2.4.20
        class_metrics.update(
            {'AvgStmtDeclNAMM': sum(number_of_stmt_decl_namm_list) / len(number_of_stmt_decl_namm_list)})
        # 2.4.21
        class_metrics.update({'AvgStmtExeNAMM': sum(number_of_stmt_exe_namm_list) / len(number_of_stmt_exe_namm_list)})

        # 2.4.22
        class_metrics.update({'MaxStmtNAMM': max(number_of_stmt_namm_list)})
        # 2.4.23
        class_metrics.update({'MaxStmtDeclNAMM': max(number_of_stmt_decl_namm_list)})
        # 2.4.24
        class_metrics.update({'MaxStmtExeNAMM': max(number_of_stmt_exe_namm_list)})

        # 2.4.25
        class_metrics.update({'MinStmtNAMM': min(number_of_stmt_namm_list)})
        # 2.4.26
        class_metrics.update({'MinStmtDeclNAMM': min(number_of_stmt_decl_namm_list)})
        # 2.4.27
        class_metrics.update({'MinStmtExeNAMM': min(number_of_stmt_exe_namm_list)})

        # 2.4.28
        class_metrics.update({'SDStmtNAMM': np.std(number_of_stmt_namm_list)})
        # 2.4.29
        class_metrics.update({'SDStmtDeclNAMM': np.std(number_of_stmt_decl_namm_list)})
        # 2.4.30
        class_metrics.update({'SDStmtExeNAMM': np.std(number_of_stmt_exe_namm_list)})

        # Class number of not accessor or mutator methods
        # Class max_nesting (4)
        CSNOMNAMM = 0
        max_nesting_list = list()
        for method in method_list:
            max_nesting_list.append(method.metric(['MaxNesting'])['MaxNesting'])
            if not j_code_odor_metric.is_accesor_or_mutator(method_entity=method):
                CSNOMNAMM += 1

        max_nesting_list = [i for i in max_nesting_list if i is not None]
        if len(max_nesting_list) == 0:
            max_nesting_list.append(0)

        class_metrics.update({'CSNOMNAMM': CSNOMNAMM})

        class_metrics.update({'MinNesting': min(max_nesting_list)})
        class_metrics.update({'AvgNesting': sum(max_nesting_list) / len(max_nesting_list)})
        class_metrics.update({'SDNesting': np.std(max_nesting_list)})

        # Custom (JCodeOdor) coupling metrics
        class_metrics.update({'RFC': j_code_odor_metric.RFC(class_name=entity)})
        class_metrics.update({'FANIN': j_code_odor_metric.FANIN(db=db, class_entity=entity)})
        class_metrics.update({'FANOUT': j_code_odor_metric.FANOUT(db=db, class_entity=entity)})

        class_metrics.update({'ATFD': UnderstandUtility.ATFD(db=db, class_entity=entity)})  ### not implement

        class_metrics.update({'CFNAMM': j_code_odor_metric.CFNAMM_Class(class_name=entity)})
        class_metrics.update({'DAC': UnderstandUtility.get_data_abstraction_coupling(db=db, class_entity=entity)})
        class_metrics.update({'NumberOfMethodCalls': UnderstandUtility.number_of_method_call(class_entity=entity)})

        # Visibility metrics
        # Understand built-in metrics plus one custom metric.
        class_metrics.update({'CSNOAMM': j_code_odor_metric.NOMAMM(class_entity=entity)})

        # Inheritance metrics
        class_metrics.update({'NIM': j_code_odor_metric.NIM(class_name=entity)})
        class_metrics.update({'NMO': j_code_odor_metric.NMO(class_name=entity)})

        class_metrics.update({'NOII': UnderstandUtility.NOII(db=db)})  ### not implement

        # class_lexicon_metrics_dict.update({'NumberOfFunctionCalls': number_of_function_calls})

        return class_metrics

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

        class_lexicon_metrics_dict.update({'NumberOfDots': dots_count})
        class_lexicon_metrics_dict.update({'NumberOfSemicolons': entity.metric(['CountSemicolon'])['CountSemicolon']})

        class_lexicon_metrics_dict.update({'NumberOfReturnAndPrintStatements': return_and_print_count})

        # print('Class lexicon metrics:', class_lexicon_metrics_dict)
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
        package_metrics = package.metric(package.metrics())
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

        classes_and_interfaces_list = UnderstandUtility.get_package_clasess_java(package_entity=package)

        # 2. Custom package metrics
        # 2.1. PKLOC (15)
        pk_loc_list = list()
        pk_loc_decl_list = list()
        pk_loc_exe_list = list()
        for type_entity in classes_and_interfaces_list:
            pk_loc_list.append(type_entity.metric(['CountLineCode'])['CountLineCode'])
            pk_loc_decl_list.append(type_entity.metric(['CountLineCodeDecl'])['CountLineCodeDecl'])
            pk_loc_exe_list.append(type_entity.metric(['CountLineCodeExe'])['CountLineCodeExe'])
        pk_loc_list = [i for i in pk_loc_list if i is not None]
        pk_loc_decl_list = [i for i in pk_loc_decl_list if i is not None]
        pk_loc_exe_list = [i for i in pk_loc_exe_list if i is not None]

        if len(pk_loc_list) == 0:
            pk_loc_list.append(0)
        if len(pk_loc_decl_list) == 0:
            pk_loc_decl_list.append(0)
        if len(pk_loc_exe_list) == 0:
            pk_loc_exe_list.append(0)

        package_metrics.update({'AvgLineCodeDecl': sum(pk_loc_decl_list) / len(pk_loc_decl_list)})
        package_metrics.update({'AvgLineCodeExe': sum(pk_loc_exe_list) / len(pk_loc_exe_list)})

        package_metrics.update({'MaxLineCode': max(pk_loc_list)})
        package_metrics.update({'MaxLineCodeDecl': max(pk_loc_decl_list)})
        package_metrics.update({'MaxLineCodeExe': max(pk_loc_exe_list)})

        package_metrics.update({'MinLineCode': min(pk_loc_list)})
        package_metrics.update({'MinLineCodeDecl': min(pk_loc_decl_list)})
        package_metrics.update({'MinLineCodeExe': min(pk_loc_exe_list)})

        package_metrics.update({'SDLineCode': np.std(pk_loc_list)})
        package_metrics.update({'SDLineCodeDecl': np.std(pk_loc_decl_list)})
        package_metrics.update({'SDLineCodeExe': np.std(pk_loc_exe_list)})

        # 2.2 PKNOS (15)
        pk_stmt_list = list()
        pk_stmt_decl_list = list()
        pk_stmt_exe_list = list()
        for type_entity in classes_and_interfaces_list:
            pk_stmt_list.append(type_entity.metric(['CountStmt'])['CountStmt'])
            pk_stmt_decl_list.append(type_entity.metric(['CountStmtDecl'])['CountStmtDecl'])
            pk_stmt_exe_list.append(type_entity.metric(['CountStmtExe'])['CountStmtExe'])
        pk_stmt_list = [i for i in pk_stmt_list if i is not None]
        pk_stmt_decl_list = [i for i in pk_stmt_decl_list if i is not None]
        pk_stmt_exe_list = [i for i in pk_stmt_exe_list if i is not None]

        if len(pk_stmt_list) == 0:
            pk_stmt_list.append(0)
        if len(pk_stmt_decl_list) == 0:
            pk_stmt_decl_list.append(0)
        if len(pk_stmt_exe_list) == 0:
            pk_stmt_exe_list.append(0)

        package_metrics.update({'AvgStmt': sum(pk_stmt_decl_list) / len(pk_stmt_decl_list)})
        package_metrics.update({'AvgStmtDecl': sum(pk_stmt_decl_list) / len(pk_stmt_decl_list)})
        package_metrics.update({'AvgStmtExe': sum(pk_stmt_exe_list) / len(pk_stmt_exe_list)})

        package_metrics.update({'MaxStmt': max(pk_stmt_list)})
        package_metrics.update({'MaxStmtDecl': max(pk_stmt_decl_list)})
        package_metrics.update({'MaxStmtExe': max(pk_stmt_exe_list)})

        package_metrics.update({'MinStmt': min(pk_stmt_list)})
        package_metrics.update({'MinStmtDecl': min(pk_stmt_decl_list)})
        package_metrics.update({'MinStmtExe': min(pk_stmt_exe_list)})

        package_metrics.update({'SDStmt': np.std(pk_stmt_list)})
        package_metrics.update({'SDStmtDecl': np.std(pk_stmt_decl_list)})
        package_metrics.update({'SDStmtExe': np.std(pk_stmt_exe_list)})

        # 2.3 PKCC (20)
        pk_cyclomatic_list = list()
        pk_cyclomatic_namm_list = list()

        pk_cyclomatic_strict_list = list()
        pk_cyclomatic_strict_namm_list = list()

        pk_cyclomatic_modified_list = list()
        pk_cyclomatic_modified_namm_list = list()

        pk_essential_list = list()
        pk_essential_namm_list = list()

        for type_entity in classes_and_interfaces_list:
            pk_cyclomatic_list.append(type_entity.metric(['SumCyclomatic'])['SumCyclomatic'])
            pk_cyclomatic_modified_list.append(type_entity.metric(['SumCyclomaticModified'])['SumCyclomaticModified'])
            pk_cyclomatic_strict_list.append(type_entity.metric(['SumCyclomaticStrict'])['SumCyclomaticStrict'])
            pk_essential_list.append(type_entity.metric(['SumEssential'])['SumEssential'])

        pk_cyclomatic_list = [i for i in pk_cyclomatic_list if i is not None]
        pk_cyclomatic_modified_list = [i for i in pk_cyclomatic_modified_list if i is not None]
        pk_cyclomatic_strict_list = [i for i in pk_cyclomatic_strict_list if i is not None]
        pk_essential_list = [i for i in pk_essential_list if i is not None]

        if len(pk_cyclomatic_list) == 0:
            pk_cyclomatic_list.append(0)
        if len(pk_cyclomatic_modified_list) == 0:
            pk_cyclomatic_modified_list.append(0)
        if len(pk_cyclomatic_strict_list) == 0:
            pk_cyclomatic_strict_list.append(0)
        if len(pk_essential_list) == 0:
            pk_essential_list.append(0)

        package_metrics.update({'MinCyclomatic': min(pk_cyclomatic_list)})
        package_metrics.update({'MinCyclomaticModified': min(pk_cyclomatic_modified_list)})
        package_metrics.update({'MinCyclomaticStrict': min(pk_cyclomatic_strict_list)})
        package_metrics.update({'MinEssential': min(pk_essential_list)})

        package_metrics.update({'SDCyclomatic': np.std(pk_cyclomatic_list)})
        package_metrics.update({'SDCyclomaticModified': np.std(pk_cyclomatic_modified_list)})
        package_metrics.update({'SDCyclomaticStrict': np.std(pk_cyclomatic_strict_list)})
        package_metrics.update({'SDEssential': np.std(pk_essential_list)})

        # 2.4 PKNESTING (4)
        pk_nesting_list = list()
        for type_entity in classes_and_interfaces_list:
            pk_nesting_list.append(type_entity.metric(['MaxNesting'])['MaxNesting'])
        pk_nesting_list = [i for i in pk_nesting_list if i is not None]  # Remove None values
        if len(pk_nesting_list) == 0:
            pk_nesting_list.append(0)

        package_metrics.update({'MinNesting': min(pk_nesting_list)})
        package_metrics.update({'AvgNesting': sum(pk_nesting_list) / len(pk_nesting_list)})
        package_metrics.update({'SDNesting': np.std(pk_nesting_list)})

        # 2.5
        # Other Size/Count metrics (understand built-in metrics)

        # PKNOMNAMM: Package number of not accessor or mutator methods
        j_code_odor = JCodeOdorMetric()
        pk_not_accessor_and_mutator_methods_list = list()
        pk_accessor_and_mutator_methods_list = list()
        for type_entity in classes_and_interfaces_list:
            pk_not_accessor_and_mutator_methods_list.append(j_code_odor.NOMNAMM(type_entity))
            pk_accessor_and_mutator_methods_list.append(j_code_odor.NOMAMM(type_entity))
        pk_not_accessor_and_mutator_methods_list = [i for i in pk_not_accessor_and_mutator_methods_list if i is not None]
        pk_accessor_and_mutator_methods_list = [i for i in pk_accessor_and_mutator_methods_list if i is not None]

        if len(pk_not_accessor_and_mutator_methods_list) == 0:
            pk_not_accessor_and_mutator_methods_list.append(0)
        if len(pk_accessor_and_mutator_methods_list) == 0:
            pk_accessor_and_mutator_methods_list.append(0)

        package_metrics.update({'PKNOMNAMM': sum(pk_not_accessor_and_mutator_methods_list)})

        # 2.6 Visibility metrics
        # Other Visibility metrics metrics (understand built-in metrics)
        package_metrics.update({'PKNOAMM': sum(pk_accessor_and_mutator_methods_list)})
        # To add other visibility metrics

        # 2.7 Inheritance metrics
        package_metrics.update({'PKNOI': len(UnderstandUtility.get_package_interfaces_java(package_entity=package))})
        package_metrics.update(
            {'PKNOAC': len(UnderstandUtility.get_package_abstract_class_java(package_entity=package))})

        return package_metrics

    @classmethod
    def compute_java_project_metrics(cls, db):
        project_metrics = db.metric(db.metrics())
        # print('number of metrics:', len(project_metrics),  project_metrics)
        # for i, metric in enumerate( project_metrics.keys()):
        #     print(i + 1, ': ',  metric,  project_metrics[metric])

        # print(project_metrics)  # Print Understand built-in metrics

        # 2 Custom project metrics
        files = UnderstandUtility.get_project_files_java(db=db)
        # 2.1 PJLOC (30)
        pj_loc_list = list()
        pj_loc_decl_list = list()
        pj_loc_exe_list = list()

        pj_stmt_list = list()
        pj_stmt_decl_list = list()
        pj_stmt_exe_list = list()

        for file_entity in files:
            pj_loc_list.append(file_entity.metric(['CountLineCode'])['CountLineCode'])
            pj_loc_decl_list.append(file_entity.metric(['CountLineCodeDecl'])['CountLineCodeDecl'])
            pj_loc_exe_list.append(file_entity.metric(['CountLineCodeExe'])['CountLineCodeExe'])

            pj_stmt_list.append(file_entity.metric(['CountStmt'])['CountStmt'])
            pj_stmt_decl_list.append(file_entity.metric(['CountStmtDecl'])['CountStmtDecl'])
            pj_stmt_exe_list.append(file_entity.metric(['CountStmtExe'])['CountStmtExe'])

        pj_loc_list = [i for i in pj_loc_list if i is not None]
        pj_loc_decl_list = [i for i in pj_loc_decl_list if i is not None]
        pj_loc_exe_list = [i for i in pj_loc_exe_list if i is not None]

        pj_stmt_list = [i for i in pj_stmt_list if i is not None]
        pj_stmt_decl_list = [i for i in pj_stmt_decl_list if i is not None]
        pj_stmt_exe_list = [i for i in pj_stmt_exe_list if i is not None]

        if len(pj_loc_list) == 0:
            pj_loc_list.append(0)
        if len(pj_loc_decl_list) == 0:
            pj_loc_decl_list.append(0)
        if len(pj_loc_exe_list) == 0:
            pj_loc_exe_list.append(0)
        if len(pj_stmt_list) == 0:
            pj_stmt_list.append(0)
        if len(pj_stmt_decl_list)== 0:
            pj_stmt_decl_list.append(0)
        if  len(pj_stmt_exe_list) == 0:
            pj_stmt_exe_list.append(0)

        project_metrics.update({'AvgLineCodeDecl': sum(pj_loc_decl_list) / len(pj_loc_decl_list)})
        project_metrics.update({'AvgLineCodeExe': sum(pj_loc_exe_list) / len(pj_loc_exe_list)})

        project_metrics.update({'MaxLineCode': max(pj_loc_list)})
        project_metrics.update({'MaxLineCodeDecl': max(pj_loc_decl_list)})
        project_metrics.update({'MaxLineCodeExe': max(pj_loc_exe_list)})

        project_metrics.update({'MinLineCode': min(pj_loc_list)})
        project_metrics.update({'MinLineCodeDecl': min(pj_loc_decl_list)})
        project_metrics.update({'MinLineCodeExe': min(pj_loc_exe_list)})

        project_metrics.update({'SDLineCode': np.std(pj_loc_list)})
        project_metrics.update({'SDLineCodeDecl': np.std(pj_loc_decl_list)})
        project_metrics.update({'SDLineCodeExe': np.std(pj_loc_exe_list)})

        # 2.2. PJNOST (15)
        project_metrics.update({'AvgStmt': sum(pj_stmt_list) / len(pj_stmt_list)})
        project_metrics.update({'AvgStmtDecl': sum(pj_stmt_decl_list) / len(pj_stmt_decl_list)})
        project_metrics.update({'AvgStmtExe': sum(pj_stmt_exe_list) / len(pj_stmt_exe_list)})

        project_metrics.update({'MaxStmt': max(pj_stmt_list)})
        project_metrics.update({'MaxStmtDecl': max(pj_stmt_decl_list)})
        project_metrics.update({'MaxStmtExe': max(pj_stmt_exe_list)})

        project_metrics.update({'MinStmt': min(pj_stmt_list)})
        project_metrics.update({'MinStmtDecl': min(pj_stmt_decl_list)})
        project_metrics.update({'MinStmtExe': min(pj_stmt_exe_list)})

        project_metrics.update({'SDStmt': np.std(pj_stmt_list)})
        project_metrics.update({'SDStmtDecl': np.std(pj_stmt_decl_list)})
        project_metrics.update({'SDStmtExe': np.std(pj_stmt_exe_list)})

        # 2.3 Other Count/Size metrics
        packages = db.ents('Java Package')
        # print('number of packages', len(packages))
        project_metrics.update({'NumberOfPackages': len(packages)})

        j_code_odor = JCodeOdorMetric()
        pj_number_of_method_namm = 0
        for class_ in UnderstandUtility.get_project_classes_java(db=db):
            pj_number_of_method_namm += j_code_odor.NOMNAMM(class_)
        project_metrics.update({'PJNOMNAMM': pj_number_of_method_namm})

        # 2.4 PJCC (20): Project cyclomatic complexity
        pj_cyclomatic_list = list()
        pj_cyclomatic_namm_list = list()

        pj_cyclomatic_strict_list = list()
        pj_cyclomatic_strict_namm_list = list()

        pj_cyclomatic_modified_list = list()
        pj_cyclomatic_modified_namm_list = list()

        pj_essential_list = list()
        pj_essential_namm_list = list()

        for type_entity in files:
            pj_cyclomatic_list.append(type_entity.metric(['SumCyclomatic'])['SumCyclomatic'])
            pj_cyclomatic_modified_list.append(type_entity.metric(['SumCyclomaticModified'])['SumCyclomaticModified'])
            pj_cyclomatic_strict_list.append(type_entity.metric(['SumCyclomaticStrict'])['SumCyclomaticStrict'])
            pj_essential_list.append(type_entity.metric(['SumEssential'])['SumEssential'])

        pj_cyclomatic_list = [i for i in pj_cyclomatic_list if i is not None]
        pj_cyclomatic_modified_list = [i for i in pj_cyclomatic_modified_list if i is not None]
        pj_cyclomatic_strict_list = [i for i in pj_cyclomatic_strict_list if i is not None]
        pj_essential_list = [i for i in pj_essential_list if i is not None]

        if len(pj_cyclomatic_list) == 0:
            pj_cyclomatic_list.append(0)
        if len(pj_cyclomatic_modified_list) == 0:
            pj_cyclomatic_modified_list.append(0)
        if len(pj_cyclomatic_strict_list) == 0:
            pj_cyclomatic_strict_list.append(0)
        if len( pj_essential_list) == 0:
            pj_essential_list.append(0)

        project_metrics.update({'SumCyclomatic': sum(pj_cyclomatic_list)})
        project_metrics.update({'SumCyclomaticModified': sum(pj_cyclomatic_modified_list)})
        project_metrics.update({'SumCyclomaticStrict': sum(pj_cyclomatic_strict_list)})
        project_metrics.update({'SumEssential': sum(pj_essential_list)})

        project_metrics.update({'MaxCyclomatic': max(pj_cyclomatic_list)})
        project_metrics.update({'MaxCyclomaticModified': max(pj_cyclomatic_modified_list)})
        project_metrics.update({'MaxCyclomaticStrict': max(pj_cyclomatic_strict_list)})
        project_metrics.update({'MaxEssential': max(pj_essential_list)})

        project_metrics.update({'AvgCyclomatic': sum(pj_cyclomatic_list) / len(pj_cyclomatic_list)})
        project_metrics.update(
            {'AvgCyclomaticModified': sum(pj_cyclomatic_modified_list) / len(pj_cyclomatic_modified_list)})
        project_metrics.update({'AvgCyclomaticStrict': sum(pj_cyclomatic_strict_list) / len(pj_cyclomatic_strict_list)})
        project_metrics.update({'AvgEssential': sum(pj_essential_list) / len(pj_essential_list)})

        project_metrics.update({'MinCyclomatic': min(pj_cyclomatic_list)})
        project_metrics.update({'MinCyclomaticModified': min(pj_cyclomatic_modified_list)})
        project_metrics.update({'MinCyclomaticStrict': min(pj_cyclomatic_strict_list)})
        project_metrics.update({'MinEssential': min(pj_essential_list)})

        project_metrics.update({'SDCyclomatic': np.std(pj_cyclomatic_list)})
        project_metrics.update({'SDCyclomaticModified': np.std(pj_cyclomatic_modified_list)})
        project_metrics.update({'SDCyclomaticStrict': np.std(pj_cyclomatic_strict_list)})
        project_metrics.update({'SDEssential': np.std(pj_essential_list)})

        # 2.4 PKNESTING (4)
        pj_nesting_list = list()
        for type_entity in files:
            pj_nesting_list.append(type_entity.metric(['MaxNesting'])['MaxNesting'])

        pj_nesting_list = [i for i in pj_nesting_list if i is not None]  # Remove None values

        if len(pj_essential_list) == 0:
            pj_nesting_list.append(0)

        project_metrics.update({'MinNesting': min(pj_nesting_list)})
        project_metrics.update({'AvgNesting': sum(pj_nesting_list) / len(pj_nesting_list)})
        project_metrics.update({'SDNesting': np.std(pj_nesting_list)})

        # 3 Inheritance metrics
        project_metrics.update({'PJNOI': len(UnderstandUtility.get_project_interfaces_java(db=db))})
        project_metrics.update({'PJNAC': len(UnderstandUtility.get_project_abstract_classes_java(db=db))})

        return project_metrics

    @classmethod
    def get_entity_kind(cls, db, class_name):
        entity = db.lookup(class_name + '$', 'Type')
        return entity[0].kindname()


# ------------------------------------------------------------------------
class PreProcess:
    """

    """
    # -------------------------------------------
    # Dataset creation API
    @classmethod
    def create_understand_database_from_project(cls, root_path=None):
        # First path
        # root_path = 'E:/LSSDS/EvoSuite/SF110-20130704-src/SF110-20130704-src/'

        # Second path, after eliminating all test class form SF110
        root_path = 'sf110_without_test/'  # A place for both project sources and understand databases

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
                    branch_coverage_dict[long_class_name].append(float(items[7]))  ## BUG!!! items[7] --> items[8]
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

        f = open('refactored_csvs/' + project_name + '.csv', mode='w', encoding='utf8')
        f.write('Class,')
        metrics_name_list = TestabilityMetrics.get_all_metrics_names()
        for metric_name in metrics_name_list:
            f.write(metric_name + ',')
        f.write('Label_LineCoverage,Label_BranchCoverage,Label_MutationScore\n')

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
    def add_one_metric_to_csv(cls, udbs_path, current_csv_dir_path):
        files = [f for f in os.listdir(udbs_path) if os.path.isfile(os.path.join(udbs_path, f))]
        for f in files:
            print('processing file {0}:'.format(f))
            db = understand.open(os.path.join(udbs_path, f))
            df = pd.read_csv(os.path.join(current_csv_dir_path, f[:-4] + 'csv'),
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

    @classmethod
    def intersection(cls, list1, list2):
        return list(set(list1) & set(list2))

    @classmethod
    def extract_coverage_before_and_after_refactoring(cls, path_before: str = None, path_after: str = None):
        df_before = pd.read_csv(path_before, delimiter=',', index_col=False, encoding='utf8',)
        df_after = pd.read_csv(path_after, delimiter=',', index_col=False, encoding='utf8')
        df = pd.DataFrame()
        df['Class'] = df_before['TARGET_CLASS']
        # quit()
        df['CoverageBeforeRefactor'] = df_before['Coverage']
        coverage_after_list = list()

        for i, class_ in enumerate(df['Class']):
            row = df_after.loc[df_after['TARGET_CLASS'] == str(class_)]
            # print(row)
            if row is None or row.empty:
                coverage_after_list.append(None)
                continue
            coverage_after_list.append(row.iloc[0]['Coverage'])
            print('{0}, class_: {1}, coverage_after: {2}'.format(i+2, class_, row.iloc[0]['Coverage']))
        df['CoverageAfterRefactor'] = coverage_after_list
        df.to_csv('refactors/mango_statistics_both.csv', index=False)



    # -------------------------------------------
    # Create complete dataset API
    @classmethod
    def create_complete_dataset(cls, separated_csvs_root, complete_csv_root):
        """
        This method merge all separated csv files which belongs to each project
        into one csv file for using with machine learning classifiers.

        :param separated_csvs_root:
        :param complete_csv_root:
        :return:
        """
        # separated_csvs_root = 'csvs/' # path csv data for dataset version 1
        # separated_csvs_root = 'sf110_csvs_without_test/'  # path of csv data for dataset version 2

        # complete_csv_file = 'es_complete_dataset_all_1_0_6_withtest_36col.csv'  # path of dataset 1
        # complete_csv_file = 'es_complete_dataset_all_1_0_6_without_test_93col.csv'  # path of dataset 2
        complete_csv_file = r'DS050.csv'
        complete_dataset_file = open(complete_csv_root + complete_csv_file, mode='w', encoding='utf8')

        complete_dataset_file.write('Class,')
        metrics_name = TestabilityMetrics.get_all_metrics_names()
        for metric_name in metrics_name:
            complete_dataset_file.write(metric_name + ',')
        complete_dataset_file.write('Label_LineCoverage,Label_BranchCoverage,Label_MutationScore\n')

        for filename in os.listdir(separated_csvs_root):
            with open(separated_csvs_root + filename, mode='r', encoding='utf8') as f:
                lines = f.readlines()
            complete_dataset_file.writelines(lines[1:])
        complete_dataset_file.close()

    # -------------------------------------------
    # Data preprocessing and cleaning API
    # Step 1:
    # Step 1.1 Remove zero columns
    @classmethod
    def remove_zero_column(cls, path: str = None, path_new: str = None):
        pd.set_option('display.max_rows', None, 'display.max_columns', None)
        pd.options.display.max_colwidth = 1000
        df = pd.read_csv(path,
                         delimiter=',',
                         index_col=False,
                         # usecols=[0,2]
                         )
        print(type(df))
        df.columns = [column.replace(' ', '_') for column in df.columns]
        df = df.loc[:, (df != 0).any(axis=0)]
        print(df.shape)
        # print(list(df.columns))
        columns_with_min_in_their_names = [i for i in df.columns if 'Min' in i]
        print('columns_with_min_in_their_names len:', len(columns_with_min_in_their_names))
        df2 = df.drop(columns=columns_with_min_in_their_names)
        # Print and save new dataset as csv and html
        print(df2.shape)
        df2.to_csv(path_new, index=False)

    # Step 1.2
    @classmethod
    def remove_zero_variance_column(cls, path: str = None, path_new: str = None):
        df1 = pd.read_csv(path, delimiter=',', index_col=False,)
        df = df1.iloc[:, 1:-5]
        all_cols = df.columns

        # 1. Drop low_variety_cols
        # df2 = df.loc[:, df.var() == 0.0]
        low_variety_cols = []
        for col in df.columns:
            if len(df[col].unique()) <= 5:
                df.drop(col, inplace=True, axis=1)
                low_variety_cols.append(col)
        print('low variety cols: {0}: {1}'.format(len(low_variety_cols), low_variety_cols))

        # 2. Drop low_variance_cols
        low_variance_cols = []
        for col in df.columns:
            # print(df[col].var())
            if df[col].var() <= 1.0:
                df.drop(col, inplace=True, axis=1)
                low_variance_cols.append(col)
        print('low_variance_cols: {0}: {1}'.format(len(low_variance_cols), low_variance_cols))

        # 3. Drop high_variance_cols
        high_variance_cols = []
        for col in df.columns:
            # print(df[col].var())
            if df[col].var() >= 1e6:
                df.drop(col, inplace=True, axis=1)
                high_variance_cols.append(col)
        print('high_variance_cols : {0}: {1}'.format(len(high_variance_cols), high_variance_cols))
        # quit()

        # 4. Drop many_zero_cols
        many_zero_cols = []
        for col in df.columns:
            # print(df[col].var())
            # print((df[col] == 0).sum(), len(df.index))
            if (df[col] == 0).sum() >= round(len(df.index)*3/4.):
                df.drop(col, inplace=True, axis=1)
                many_zero_cols.append(col)
        print('many_zero_cols: {0}: {1}'.format(len(many_zero_cols), many_zero_cols))
        print(df.shape)

        df.insert(loc=0, column='Class', value=df1['Class'])
        df['Label_LineCoverage'] = df1['Label_LineCoverage']
        df['Label_BranchCoverage'] = df1['Label_BranchCoverage']
        df['Label_MutationScore'] = df1['Label_MutationScore']
        df['Label_Combine1'] = df1['Label_Combine1']
        df['Label_Combine2'] = df1['Label_Combine2']
        print('Before dropping many zero rows:', df.shape)

        # 5. Drop many_zero_rows
        print('-' * 25)
        many_zero_rows = []

        for index, item in ((df == 0).sum(1)).iteritems():
            if item >= round((len(df.columns)-6) * 1/3):
                # print(index, item)
                many_zero_rows.append([index, item])
                df.drop(index=index, axis=0, inplace=True)
        print('many_zero_rows {0}: {1}'.format(len(many_zero_rows), many_zero_rows[0]))
        print('After dropping many zero rows:', df.shape)

        # 6. Statistics
        print('Total number of zeros: {0}'.format((df == 0).sum(1).sum()))
        print('Total number of non zeros: {0}'.format((df != 0).sum(1).sum()))
        print('Total number of items: {0}'.format(len(df.columns) * len(df.index)))
        print('Portion of zeros: {0}'.format(((df == 0).sum(1).sum()) / (len(df.columns)*len(df.index))))


        # non_constant_cols = df.columns
        # constant_col = (set(all_cols)).difference(set(non_constant_cols))
        # print(len(constant_col))
        # print(constant_col)
        df.to_csv(path_new, index=False)

    # Step 2: Discretization (Convert numerical branch coverage to nominal coverageability labels)
    # Step 2.1:
    @classmethod
    def discretize(cls, path: str = None, path_new: str = None):
        """
        https://pbpython.com/pandas-qcut-cut.html
        quantile_based_discretization
        :param path:
        :param path_new:
        :return:
        """
        data_frame = pd.read_csv(path,
                                 delimiter=',',
                                 index_col=False,
                                 # usecols=[0,2]
                                 )
        # data_frame.columns = [column.replace(' ', '_') for column in data_frame.columns]
        # print(data_frame)
        # quit()

        # Define fine-grain coverageability nominal labels (five category)
        coverageability_labels = ['VeryLow', 'Low', 'Mean', 'High', 'VeryHigh']
        coverageability_labels = ['Low', 'Moderate', 'High', ]
        bins = 3
        # bins = pd.IntervalIndex.from_tuples([(-0.001, 0.30), (0.30, 0.70), (0.70, 1.001)])
        # bins = [-0.01, 25.0, 75.0, 100.01]
        bins = [-0.01, 20.0, 80.0, 100.01]
        # Add coverageability column
        data_frame['CoverageabilityNominal'] = pd.cut(data_frame.loc[:, ['Label_BranchCoverage']].T.squeeze(),
                                                      bins=bins,
                                                      labels=coverageability_labels,
                                                      right=True
                                                      )
        # print(pd.cut(data.loc[:, ['_branch_coverage']].T.squeeze(),
        #              bins=5,
        #              labels=['VeryLow', 'Low', 'Mean', 'High', 'VeryHigh']
        #              ).value_counts())

        data_frame['CoverageabilityNominalCombined'] = pd.cut(data_frame.loc[:, ['Label_Combine2']].T.squeeze(),
                                                      bins=bins,
                                                      labels=coverageability_labels,
                                                      right=True
                                                      )

        testability_labels = ['NonTestable', 'Testable']
        data_frame['TestabilityNominal'] = pd.cut(data_frame.loc[:, ['Label_BranchCoverage']].T.squeeze(),
                                                  bins=2,
                                                  labels=testability_labels
                                                  )
        testability_labels_binary = [0, 1]
        data_frame['TestabilityBinary'] = pd.cut(data_frame.loc[:, ['Label_BranchCoverage']].T.squeeze(),
                                                 bins=2,
                                                 labels=testability_labels_binary
                                                 )
        print(data_frame.shape)

        # Remove extra columns
        columns_list = ['Label_LineCoverage', 'Label_BranchCoverage', 'Label_MutationScore', 'Label_Combine1', 'Label_Combine2']
        # columns_list = ['Label_LineCoverage', 'Label_MutationScore']
        data_frame_dropped = data_frame.drop(columns_list, axis=1)

        # Print and save new dataset as csv and html
        print(data_frame_dropped)
        # path_new = r'es_complete_dataset_all_1_0_6_without_test_93col_discretize_91col_15417.csv'
        # print(data_frame_dropped.shape)
        data_frame_dropped.to_csv(path_new, index=False)
        # data_frame.to_csv(path_new, index=False)

    # Step 2.2
    #  Discretize variable into equal-sized buckets based
    @classmethod
    def discretize_q(cls, path: str = None, path_new: str = None):
        """
        quantile_based_discretization
        :param path:
        :param path_new:
        :return:
        """
        data_frame = pd.read_csv(path, delimiter=',', index_col=False,)
        # data_frame['Label_BranchCoverage'].replace(to_replace=0, value=np.nan, inplace=True)

        # Define fine-grain coverageability nominal labels (five category)
        coverageability_labels = ['VeryLow', 'Low', 'Mean', 'High', 'VeryHigh']
        coverageability_labels = ['Low', 'Moderate', 'High',]
        # Add coverageability column
        data_frame['CoverageabilityNominalCombined'] = pd.qcut(data_frame.Label_Combine2,
                                                      q=3,
                                                      labels=coverageability_labels,
                                                      # duplicates='drop'
                                                       )
        # print(pd.cut(data.loc[:, ['_branch_coverage']].T.squeeze(),
        #              bins=5,
        #              labels=['VeryLow', 'Low', 'Mean', 'High', 'VeryHigh']
        #              ).value_counts())

        print(data_frame)
        # quit()

        testability_labels = ['NonTestable', 'Testable']
        data_frame['TestabilityNominal'] = pd.qcut(data_frame.Label_BranchCoverage,
                                                  q=2,
                                                  labels=testability_labels
                                                  )
        testability_labels_binary = [0, 1]
        data_frame['TestabilityBinary'] = pd.qcut(data_frame.Label_BranchCoverage,
                                                 q=2,
                                                 labels=testability_labels_binary
                                                 )
        print(data_frame.shape)

        # Remove extra columns
        columns_list = ['Label_LineCoverage', 'Label_BranchCoverage', 'Label_MutationScore', 'Label_Combine1',
                        'Label_Combine2']
        # columns_list = ['Label_LineCoverage', 'Label_MutationScore']
        data_frame_dropped = data_frame.drop(columns_list, axis=1)

        # Print and save new dataset as csv and html
        print(data_frame_dropped)
        # path_new = r'es_complete_dataset_all_1_0_6_without_test_93col_discretize_91col_15417.csv'
        # print(data_frame_dropped.shape)
        data_frame_dropped.to_csv(path_new, index=False)
        # data_frame.to_csv(path_new, index=False)

    # Step 3: Remove data classes
    @classmethod
    def mitigate_imbalanced(cls, path: str = None, path_new: str = None):
        """

        :param path: The path of complete dataset (raw data with 3 tagged column)
        :param path_new: The path of new dataset
        :return:
        """
        # pd.set_option('display.max_rows', None, 'display.max_columns', None)
        # pd.options.display.max_colwidth = 1000
        df = pd.read_csv(path, delimiter=',',
                         index_col=False,
                         # usecols=[0,2],
                         )
        # df.columns = [column.replace(' ', '_') for column in df.columns]
        print('df_:', df.shape)

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

        df1 = df.loc[((df['Label_BranchCoverage'] < 5.0) & (df.CSORD_SumCyclomatic <= 5))]

        # data1 = data1.filter(like='UnifyCase', axis=0)
        # data1 = data1[data1.ClassOrdinary_MaxNesting == data1.ClassOrdinary_MaxNesting.max()]
        # data1 = data1[data1.ClassOrdinary_AvgCyclomatic == data1.ClassOrdinary_AvgCyclomatic.max()]
        # data1 = data1[data1.ClassOrdinary_SumCyclomatic == data1.ClassOrdinary_SumCyclomatic.max()]
        # data1 = data1[data1.ClassOrdinary_CountLineCode == data1.ClassOrdinary_CountLineCode.max()]
        df1 = pd.concat([df, df1]).drop_duplicates(keep=False)

        print('df1:', df1.shape)

        # Put data classes into df2
        df2 = df1.loc[(df1['Label_BranchCoverage'] >= 100.0)
                      # & (data['ClassOrdinary_CountLineCode'] == 10e6)
                      # & (data['class'].str.contains('weka'))
                      # & (data.ClassOrdinary_MaxNesting == data.ClassOrdinary_MaxNesting.max())
                      & (df1.CSORD_SumCyclomatic <= 1)
            # & (df1.CSORD_MaxCyclomatic <= 1)
            # & (data['ClassOrdinary_CountDeclMethodAll'] <= 5)
            # & (data.ClassOrdinary_CountDeclMethod <= 0)
            # & (data.ClassOrdinary_MaxNesting == 0)
                      ]
        # Remove data classes from df1 and put result in df2
        df2 = pd.concat([df2, df1]).drop_duplicates(keep=False)
        print('df2:', df2.shape)

        # Put data classes into df3
        df3 = df2.loc[(df2['Label_BranchCoverage'] >= 100.0)
                      & (df2.CSORD_CountLineCodeExeNAMM <= 0)
                      ]
        # Remove data classes from df1 and put result in df3
        df3 = pd.concat([df3, df2]).drop_duplicates(keep=False)
        print('df3:', df3.shape)

        # Put data classes into df4
        df4 = df3.loc[(df3['Label_BranchCoverage'] >= 100.0)
                      & (df3.CSORD_SumCyclomaticNAMM <= 1)
                      ]
        # Remove data classes from df1 and put result in df2
        df4 = pd.concat([df4, df3]).drop_duplicates(keep=False)
        print('df4:', df4.shape)

        # Put data classes into df5
        df5 = df4.loc[(df4['Label_BranchCoverage'] >= 100.0)
                      & (df4.CSORD_CountLineCodeExe <= 1)
                      ]
        df5 = pd.concat([df5, df4]).drop_duplicates(keep=False)
        print('df5:', df5.shape)

        # Put data classes into df6
        df6 = df5.loc[(df5['Label_BranchCoverage'] >= 100.0)
                      & (df5.CSORD_CountDeclInstanceMethod <= 1)
                      ]
        df6 = pd.concat([df6, df5]).drop_duplicates(keep=False)
        print('df6:', df6.shape)

        # !Mitigate Zero coverages
        df7 = df6.loc[(df6['Label_BranchCoverage'] < 5.0) &
                      (df6.CSORD_CountLineCodeExe <= 5)
                      ]

        df7 = pd.concat([df7, df6]).drop_duplicates(keep=False)
        print('df7:', df7.shape)

        # !
        df8 = df7.loc[(df7['Label_BranchCoverage'] >= 100.0) &
                      (df7.CSORD_CountStmtExeNAMM <= 1)
                      ]
        df8 = pd.concat([df8, df7]).drop_duplicates(keep=False)
        print('df8:', df8.shape)

        # --
        df9 = df8.loc[(df8['Label_BranchCoverage'] >= 100.0) &
                      (df8.CSORD_CSNOMNAMM <= 0)
                      ]
        df9 = pd.concat([df9, df8]).drop_duplicates(keep=False)
        print('df9:', df9.shape)

        # Temporary code for experiment
        # df3 = df2.loc[
        #     (df['Label_BranchCoverage'] < 1.)
        #     & (df.ClassOrdinary_CountDeclMethodAll >= 0)
        #     ]
        # print(df3.shape)
        # df3 = df2.loc[:, ['Class',
        #                       'ClassOrdinary_CountLineCode',
        #                       'ClassOrdinary_AvgCyclomatic',
        #                       'ClassOrdinary_SumCyclomatic',
        #                       'ClassOrdinary_MaxNesting',
        #                       'Label_BranchCoverage']]
        # print('df3:', df3.shape)

        # col = data['ClassOrdinary_CountLineCode']
        # print('max col is in row {0} with value {1} and name {3}'.format(col.idxmax(), col.max(), col[col.idxmax(), :]))
        # print(data.max())
        # print(data[data.ClassOrdinary_CountLineCode ==
        #            data.ClassOrdinary_CountLineCode.max()][['Class', 'Label_LineCoverage']])

        # Print and save new dataset as csv and html
        # df2.to_csv(path_new,
        #            header=True,
        #            index=True,
        #            index_label='index')
        df9.to_csv(path_new, index=False)

    # Step 4: Remove outlier records based on z_scores of all features (base data preparing has finished)
    # Step 4.1: Remove outliers with z-score
    @classmethod
    def remove_outliers(cls, path: str = None, path_new: str = None):
        # pd.set_option('display.max_rows', None, 'display.max_columns', None)
        pd.options.display.max_colwidth = 1000
        df = pd.read_csv(path,
                         delimiter=',',
                         index_col=False,
                         # usecols=[0,2]
                         )
        df.columns = [column.replace(' ', '_') for column in df.columns]

        non_features_columns = ['Class',
                                'Label_BranchCoverage',
                                'CoverageabilityNominal',
                                'TestabilityNominal', 'TestabilityBinary']
        df2 = df.drop(columns=non_features_columns)

        """
        # New for version 0.3.0 (ignore in version 4)
        # We include only primary metrics set in outlier removing process
        # Remove systematically generated metrics from data frame
        p_names = set(TestabilityMetrics.get_all_primary_metrics_names())
        s_names = set(TestabilityMetrics.get_all_metrics_names())
        print('p_names', len(p_names))
        print('s_names', len(s_names))
        systematically_generated_metric_list = s_names.difference(p_names)
        print(systematically_generated_metric_list)
        print('len systematically_generated_metric_list',
              len(systematically_generated_metric_list))

        systematically_generated_metric_list = [i for i in systematically_generated_metric_list if 'Min' not in i]
        print('len systematically_generated_metric_list', len(systematically_generated_metric_list))
        df2 = df2.drop(columns=list(systematically_generated_metric_list))
        print(df2.columns)
        print(df2.shape)
        # quit()
        """

        z_scores = stats.zscore(df2)
        # print(z_scores)
        abs_z_scores = np.abs(z_scores)
        filtered_entries = (abs_z_scores < 3).all(axis=1)
        df3 = df[filtered_entries]  # we can use df or df2
        print('df3:', df3.shape)

        # Print and save new dataset as csv and html
        # path_new = r'es_complete_dataset_all_1_0_6_without_test_93col_discretize_91col_15417_outlier_removed.csv'
        df3.to_csv(path_new, index=False)

    # Step 4.2: Remove outliers with z-score
    @classmethod
    def remove_outliers_with_lof(cls, path: str = None, path_new: str = None):
        # https://machinelearningmastery.com/model-based-outlier-detection-and-removal-in-python/
        # https://scikit-learn.org/stable/auto_examples/ensemble/plot_isolation_forest.html
        # pd.set_option('display.max_rows', None, 'display.max_columns', None)
        pd.options.display.max_colwidth = 1000
        df = pd.read_csv(path, delimiter=',', index_col=False,)
        df.columns = [column.replace(' ', '_') for column in df.columns]

        X1 = df.iloc[:, 1:-5]
        y1 = df.iloc[:, -3]

        data = df.values
        X = data[:, 1:-5]
        y0 = data[:, 0]
        y1 = data[:, -5:]

        lof = LocalOutlierFactor(n_neighbors=256)
        yhat = lof.fit_predict(X)

        # select all rows that are not outliers
        mask = yhat != -1
        X, y0, y1 = X[mask, :], y0[mask], y1[mask]

        # summarize the shape of the updated training dataset
        print(X.shape, y0.shape, y1.shape)

        df2 = pd.DataFrame(X, columns=X1.columns)
        df2.insert(loc=0, column='Class', value=y0)
        df2['Label_LineCoverage'] = y1[:, 0]
        df2['Label_BranchCoverage'] = y1[:, 1]
        df2['Label_MutationScore'] = y1[:, 2]
        df2['Label_Combine1'] = y1[:, 3]
        df2['Label_Combine2'] = y1[:, 4]

        print(df2)
        df2.to_csv(path_new, index=False)



    # Step 5: Training set/ testing set split and save
    @classmethod
    def split_dataset_base(cls, path: str = None, path_new: str = None):
        df = pd.read_csv(path, delimiter=',', index_col=False)
        # X = df.iloc[:, 1:-4]
        # y = df.iloc[:, -3]
        X = df.iloc[:, 1:-3]
        y = df.iloc[:, -3]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42,
                                                            # stratify=y
                                                            )
        # print(X_train.head(), y_train.head())

        df_train = pd.DataFrame(X_train)
        df_train['CoverageabilityNominal'] = y_train
        print(df_train)

        df_test = pd.DataFrame(X_test)
        df_test['CoverageabilityNominal'] = y_test
        print(df_test)

        df_train.to_csv(path_new+'train.csv', index=False)
        df_test.to_csv(path_new+'test.csv', index=False)

    # -------------------------------------------
    # Step 6: Resampling
    # Step 6.1
    @classmethod
    def resampling(cls, path: str = None, path_new: str = None):
        # pd.set_option("display.max_rows", None, "display.max_columns", None)
        pd.options.display.max_colwidth = 1000
        df = pd.read_csv(path, delimiter=',', index_col=False,)
        print('Before resampling:', df.shape)
        # Strategy one: Simply duplicate the data!: Dose not work on test set
        df2 = df.append(df, ignore_index=True)

        # Strategy two: Combination of over- and under-sampling
        # https://imbalanced-learn.readthedocs.io/en/stable/combine.html
        # X = df.iloc[:, 1: -4]
        # y = df.iloc[:, -3]

        X = df.iloc[:, : -1]
        y = df.iloc[:, -1]

        # X_resampled, y_resampled = SMOTE(random_state=42, k_neighbors=10).fit_resample(X, y)
        # X_resampled, y_resampled = ADASYN(sampling_strategy='auto', random_state=42, n_neighbors=100, ).fit_resample(X, y)

        # smote_tomek = SMOTETomek(random_state=42, )
        # X_resampled, y_resampled = smote_tomek.fit_resample(X, y)

        # SMOTEENN tends to clean more noisy samples than SMOTETomek.
        smote_enn = SMOTEENN(random_state=42, )
        X_resampled, y_resampled = smote_enn.fit_resample(X, y)

        print(sorted(Counter(y_resampled).items()))
        df2 = pd.DataFrame(X_resampled)
        # df2.insert(loc=0, column='Class', value=df['Class'])
        df2['CoverageabilityNominal'] = y_resampled

        # Print and save new dataset as csv and html
        # path_new = r'es_complete_dataset_all_1_0_6_without_test_93col_discretize_91col_15417_outlier_removed_binary_SMOTEENN.csv'
        print('After resampling', df2.shape)
        df2.to_csv(path_new, index=False)

    #  Step 6.2
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
        X = df2.iloc[:, 0: -1]
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

    # Step 7: Normalization
    @classmethod
    def normalize_features(cls, path: str = None, new_path: str = None):
        """
        https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html#sphx-glr-auto-examples-preprocessing-plot-all-scaling-py
        :param path:
        :param new_path:
        :return:
        """
        # pd.set_option("display.max_rows", None, "display.max_columns", None)
        pd.options.display.max_colwidth = 1000
        df = pd.read_csv(path, delimiter=',', index_col=False)
        df.columns = [column.replace(' ', '_') for column in df.columns]

        non_features_columns = ['Class',
                                'Label_BranchCoverage',
                                'CoverageabilityNominal',
                                'TestabilityNominal', 'TestabilityBinary']
        # df2 = df.drop(columns=non_features_columns)
        # x = df2.values  # returns a numpy array
        df2 = df.iloc[:, :-1]
        x = df2.values

        # Standardization 1
        std_scaler = preprocessing.StandardScaler(with_mean=True, with_std=True)
        x_scaled = std_scaler.fit_transform(x)

        # Standardization 2
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        # x_scaled = min_max_scaler.fit_transform(x)

        # Standardization 3
        robust_scaler = preprocessing.RobustScaler(with_centering=True, with_scaling=True)
        # x_scaled = robust_scaler.fit_transform(x)

        # Normalization 1
        normalizer = preprocessing.Normalizer(norm='l2')
        # x_scaled = normalizer.fit_transform(x)

        df3 = pd.DataFrame(x_scaled)


        # df3.insert(loc=0, column='Class', value=df['Class'])
        # df3.insert(loc=-4, column='Label_BranchCoverage', value=df['Label_BranchCoverage'])
        # df3.insert(loc=-3, column='Label_BranchCoverage', value=df['Label_BranchCoverage'])
        # df3['Label_BranchCoverage'] = df['Label_BranchCoverage']
        df3['CoverageabilityNominal'] = df['CoverageabilityNominal']
        # df3['TestabilityNominal'] = df['TestabilityNominal']
        # df3['TestabilityBinary'] = df['TestabilityBinary']
        df3.columns = df.columns

        print('df3_normalized:', df3.shape)
        # Print and save new dataset as csv and html
        # new_path = r'es_complete_dataset_all_1_0_6_without_test_93col_discretize_91col_15417_outlier_removed_min_max_scaler.csv'
        df3.to_csv(new_path, index=False)

    # Step 8: Feature Engineering
    # Step 8.1: Feature selection
    @classmethod
    def select_feature(cls, path: str = None, path_new: str = None):
        df = pd.read_csv(path,
                         delimiter=',',
                         index_col=False,
                         # usecols=[0,2]
                         )
        print('df shape:', df.shape)
        # X = df.iloc[:, 1:-4]  # independent columns
        # y = df.iloc[:, -3]  # target column i.e coverageability

        X = df.iloc[:, :-1]  # independent columns
        y = df.iloc[:, -1]  # target column i.e coverageability

        # Apply SelectKBest class to extract top 10 best features
        # X_new = SelectKBest(chi2, k=10).fit_transform(X, y)
        # X_new = SelectKBest(chi2, k=30).fit(X, y)
        # df2 = pd.DataFrame(X_new)

        selector = SelectKBest(f_classif, k=20)
        selector.fit(X, y)

        # Get columns to keep and create new dataframe with those only
        cols = selector.get_support(indices=True)
        df2 = X.iloc[:, cols]
        # df2.insert(loc=0, column='Class', value=df['Class'])
        # df2['Label_BranchCoverage'] = df['Label_BranchCoverage']
        df2.loc[:, 'CoverageabilityNominal'] = y
        # df2['TestabilityNominal'] = df['TestabilityNominal']
        # df2['TestabilityBinary'] = df['TestabilityBinary']

        print('df2 shape: ', df2.shape)
        df2.to_csv(path_new, index=False)
        # df2.to_html(path_new+'.html', index=False)

    # Step 8.2: Feature extraction
    @classmethod
    def extract_feature(cls, path: str = None, path_new: str = None, number_of_features=2):
        df = pd.read_csv(path, delimiter=',', index_col=False,)
        print('df shape:', df.shape)

        X = df.iloc[:, 0:-1]  # independent columns
        y = df.iloc[:, -1]  # target column i.e coverageability

        pca = PCA(n_components=number_of_features)
        pca.fit(np.array(X))
        vectors_2d = pca.transform(np.array(X))

        # print(vectors_2d)
        # print(len(vectors_2d))

        df2 = pd.DataFrame(vectors_2d, columns=['F' + str(i) for i in range(1, number_of_features+1)])
        # df2.insert(loc=0, column='Class', value=df['Class'])
        # df2['Label_BranchCoverage'] = df['Label_BranchCoverage']
        df2['CoverageabilityNominal'] = y
        # df2['TestabilityNominal'] = df['TestabilityNominal']
        # df2['TestabilityBinary'] = df['TestabilityBinary']

        print(df2.shape)
        df2.to_csv(path_new, index=False)

    @classmethod
    def select_feature_for_testing_set(cls, path_training_set: str = None, path: str = None, path_new: str = None, ):
        df_train = pd.read_csv( path_training_set, delimiter=',', index_col=False)
        df_test = pd.read_csv(path, delimiter=',',index_col=False)

        df_test_selected_feature = pd.DataFrame()
        for col in df_train.columns:
            df_test_selected_feature[col] = df_test[col]

        df_test_selected_feature.to_csv(path_new, index=False)


    # -------------------------------------------
    # Remove context vector
    @classmethod
    def remove_context_vector(cls, path: str = None, path_new: str = None):
        df = pd.read_csv(path,
                         delimiter=',',
                         index_col=False,
                         # usecols=[0,2]
                         )
        print('df shape', df.shape)

        context_vector_names = [i for i in df.columns if 'PJ_' in i]
        context_vector_names.extend([i for i in df.columns if 'PK_' in i])
        print('len context vector', len(context_vector_names))

        df = df.drop(columns=context_vector_names)
        print(df.shape)
        df.to_csv(path_new, index=False)

    @classmethod
    def remove_context_vector_and_lexicon_metrics(cls, path: str = None, path_new: str = None):
        df = pd.read_csv(path,
                         delimiter=',',
                         index_col=False,
                         # usecols=[0,2]
                         )
        print('df shape', df.shape)

        context_vector_names = [i for i in df.columns if 'PJ_' in i]
        context_vector_names.extend([i for i in df.columns if 'PK_' in i])
        context_vector_names.extend([i for i in df.columns if 'CSLEX_' in i])

        print('len context vector and lexicon metrics', len(context_vector_names))

        df = df.drop(columns=context_vector_names)
        print(df.shape)
        df.to_csv(path_new, index=False)

    @classmethod
    def remove_systematically_generated_metrics(cls, path: str = None, path_new: str = None):
        df = pd.read_csv(path,
                         delimiter=',',
                         index_col=False,
                         # usecols=[0,2]
                         )

        print('df shape', df.shape)

        # New for version 0.3.0
        # We include only primary metrics set in outlier removing process
        # Remove systematically generated metrics from data frame
        p_names = set(TestabilityMetrics.get_all_primary_metrics_names())
        s_names = set(TestabilityMetrics.get_all_metrics_names())
        print('p_names', len(p_names))
        print('s_names', len(s_names))
        systematically_generated_metric_list = s_names.difference(p_names)
        print(systematically_generated_metric_list)
        print('len systematically_generated_metric_list',
              len(systematically_generated_metric_list))

        systematically_generated_metric_list = [i for i in systematically_generated_metric_list if 'Min' not in i]
        print('len systematically_generated_metric_list',
              len(systematically_generated_metric_list))

        df = df.drop(columns=list(systematically_generated_metric_list))

        print(df.shape)
        df.to_csv(path_new, index=False)


    # -------------------------------------------
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
        testability_labels = {'VeryLow': 'NonTestable', 'Low': 'NonTestable', 'Mean': 'NonTestable', 'High': 'Testable',
                              'VeryHigh': 'Testable'}
        df.Testability = [testability_labels[item] for item in df.Testability]

        print(df)
        df.to_csv(r'es_complete_dataset_all_1_0_6_without_test_93col_discretize_91col_15417_outlier_removed_binary.csv',
                  index=False)