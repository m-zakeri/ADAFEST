"""
The main function for testability measurement modules.

"""
__version__ = '0.1.5'
__author__ = 'Morteza'

# from testability.data_preparation_evo_suite_2 import *
from adafest.code.preprocessing.data_preparation_evosuite_3 import *


def test_main_driver():
    # Data path
    dataset_path = r'es_compressed_data_all_1_0_6'  # EvoSuite dataset
    dataset_path = r'es_compressed_data_all_1_0_6_refactored'  # Experiment RQ7 dataset

    # database_path = 'udbs/'
    database_path = 'sf110_without_test/'  # new udbs
    # understab_db_path = 'udbs/15_beanbin.udb'
    # understab_db_path = 'sf110_without_test/5_templateit.udb'

    # Test operations
    # db = understand.open(understab_db_path)
    # TestabilityMetrics.compute_java_class_metrics_lexicon(db=db,
    #                                                       class_name='net.sourceforge.beanbin.codegen.BlobClassMaker')
    # TestabilityMetrics.compute_java_package_metrics(db=db, class_name='net.sourceforge.beanbin.codegen.BlobClassMaker')
    # TestabilityMetrics.compute_java_project_metrics(db=db)
    # UnderstandUtility.get_db_tokens(db, look_up_string=".\.java")

    # database_path = 'udbs/82_gaj.udb'
    # database_path = 'udbs/17_inspirento.udb'

    # db = understand.open(database_path)
    # print(UnderstandUtility.get_java_class_names(db))
    # PreProcess.extract_metrics_and_coverage(dataset=dataset_path,
    #                                         database=db,
    #                                         project_name='82_gaj')

    # classes_list = TestabilityMetrics.extract_class(db=db)
    # intersection = PreProcess.intersection(classes_list, classes_coverage_list)
    # print('Intersection', len(intersection), intersection)
    # TestabilityMetrics.compute_java_class_metrics(db=db,
    #                                               class_name='net.sourceforge.squirrel_sql.client.util.codereformat.CodeReformatorKernel')

    # entity = db.lookup('net.sourceforge.squirrel_sql.fw.dialects.DialectType' + '$', 'Type')
    # print('#', len(entity), entity)
    # print(entity[0].kindname())

    # metrics_names_list = TestabilityMetrics.get_all_metrics_mnames()
    # print('metrics_names_list: ', metrics_names_list)
    # print('number of metrics: ', len(metrics_names_list))
    # print('-' * 75)


def process_dataset030():
    # DS-0.3.0
    # PreProcess.alleviate_imbalanced(path=r'es_complete_dataset_all_1_0_6_without_test_93col.csv')
    # PreProcess.quantile_based_discretization(path=r'es_complete_dataset_all_1_0_6_without_test_93col_15417.csv')
    # PreProcess.removing_outliers(path=r'es_complete_dataset_all_1_0_6_without_test_93col_discretize_91col_15417.csv')
    # PreProcess.normalize_features(path=r'es_complete_dataset_all_1_0_6_without_test_93col_discretize_91col_15417_outlier_removed.csv')
    # PreProcess.resampling(path=r'es_complete_dataset_all_1_0_6_without_test_93col_discretize_91col_15417_outlier_removed_binary.csv')
    # PreProcess.resampling_numerical_dataset(r'es_complete_dataset_all_1_0_6_without_test_93col_15417.csv')
    # PreProcess.add_testability_values(r'es_complete_dataset_all_1_0_6_without_test_93col_discretize_91col_15417_outlier_removed_11002.csv')

    print('-' * 75)
    print('STEP 5: Clean dataset version 0.3.0')

    # DS-0.3.0
    # PreProcess.discretize(path=r'dataset03/DS030.csv', path_new=r'dataset03/DS0301.csv')
    # PreProcess.mitigate_imbalanced(path=r'dataset03/DS0301.csv', path_new=r'dataset03/DS03012.csv')
    # PreProcess.remove_zero_column(path=r'dataset03/DS03012.csv', path_new=r'dataset03/DS03013.csv')
    # PreProcess.remove_outliers(path=r'dataset03/DS03013.csv', path_new=r'dataset03/DS03014.csv')

    # DS-0.3.1
    # PreProcess.resampling(path=r'dataset03/DS03014.csv', path_new=r'dataset03/DS03100.csv')

    # DS-0.3.2
    # PreProcess.normalize_features(path=r'dataset03/DS03014.csv', new_path=r'dataset03/DS03200.csv')
    # PreProcess.resampling(path=r'dataset03/DS03200.csv', path_new=r'dataset03/DS03201.csv')
    # PreProcess.extract_feature(path=r'dataset03/DS03200.csv', path_new=r'dataset03/DS03202.csv', number_of_features=10)

    # DS-0.3.3
    # PreProcess.select_feature(path=r'dataset03/DS03014.csv', path_new=r'dataset03/DS03300.csv')
    # PreProcess.resampling(path=r'dataset03/DS03300.csv', path_new=r'dataset03/DS03301.csv')

    # DS-0.3.4
    # PreProcess.normalize_features(path=r'dataset03/DS03014.csv', new_path=r'dataset03/DS03400.csv')
    # PreProcess.select_feature(path=r'dataset03/DS03400.csv', path_new=r'dataset03/DS03401.csv')
    # PreProcess.resampling(path=r'dataset03/DS03401.csv', path_new=r'dataset03/DS03402.csv')

    # DS-0.3.5
    # PreProcess.remove_context_vector(path=r'dataset03/DS03014.csv', path_new=r'dataset03/DS03500.csv')
    # PreProcess.resampling(path=r'dataset03/DS03500.csv', path_new=r'dataset03/DS03501.csv')

    # DS-0.3.6
    # PreProcess.remove_context_vector_and_lexicon_metrics(path=r'dataset03/DS03014.csv', path_new=r'dataset03/DS03600.csv')
    # PreProcess.resampling(path=r'dataset03/DS03600.csv', path_new=r'dataset03/DS03601.csv')

    # DS-0.3.7
    # PreProcess.remove_systematically_generated_metrics(path=r'dataset03/DS03014.csv',
    #                                                      path_new=r'dataset03/DS03700.csv')
    # PreProcess.resampling(path=r'dataset03/DS03700.csv', path_new=r'dataset03/DS03701.csv')

    # -------------------------------------------
    # PreProcess.extract_coverage_before_and_after_refactoring('refactors/mango_statistics_before.csv',
    #                                                          'refactors/mango_statistics_after.csv')


def process_dataset040():
    # DS-0.4.0
    # Start with DS040 (Corrected columns names)
    PreProcess.remove_zero_variance_column(path=r'dataset04/DS040.csv', path_new=r'dataset04/DS04010.csv')
    # PreProcess.discretize(path=r'dataset04/DS04031.csv', path_new=r'dataset04/DS04024.csv')
    # PreProcess.discretize_q(path=r'dataset04/DS04031.csv', path_new=r'dataset04/DS04022.csv')

    # PreProcess.mitigate_imbalanced(path=r'dataset04/DS04020.csv', path_new=r'dataset04/DS04031.csv')
    # PreProcess.remove_outliers(path=r'dataset04/DS04031.csv', path_new=r'dataset04/DS04041.csv')   # !DS base
    # PreProcess.remove_outliers_with_lof(path=r'dataset04/DS04024.csv', path_new=r'dataset04/DS04046.csv')  # !DS base
    # PreProcess.remove_outliers(path=r'dataset04/DS04020.csv', path_new=r'dataset04/DS04050.csv')
    # PreProcess.split_dataset_base(path=r'dataset04/DS04046.csv', path_new=r'dataset04/DS04046_')

    # DS-0.4.1
    # PreProcess.resampling(path=r'dataset04/DS04046_train.csv', path_new=r'dataset04/DS04146_train.csv')
    # PreProcess.resampling(path=r'dataset04/DS04050.csv', path_new=r'dataset04/DS04120.csv')

    # DS-0.4.2
    # PreProcess.normalize_features(path=r'dataset03/DS04040.csv', new_path=r'dataset03/DS04210.csv')
    # PreProcess.resampling(path=r'dataset03/DS03200.csv', path_new=r'dataset03/DS03201.csv')
    # PreProcess.extract_feature(path=r'dataset03/DS03200.csv', path_new=r'dataset03/DS03202.csv', number_of_features=10)

    # DS-0.4.3
    # PreProcess.select_feature(path=r'dataset04/DS04040s_train.csv', path_new=r'dataset04/DS04310s_train.csv')
    # PreProcess.select_feature_for_testing_set(path_training_set=r'dataset04/DS04310s_train.csv',
    #                                           path=r'dataset04/DS04040s_test.csv',
    #                                           path_new=r'dataset04/DS04310s_test.csv')
    # PreProcess.resampling(path=r'dataset03/DS03300.csv', path_new=r'dataset03/DS03301.csv')

    # DS-0.4.4
    # PreProcess.normalize_features(path=r'dataset04/DS04046_train.csv', new_path=r'dataset04/DS04420_train.csv')
    # PreProcess.normalize_features(path=r'dataset04/DS04046_test.csv', new_path=r'dataset04/DS04420_test.csv')
    # PreProcess.select_feature(path=r'dataset04/DS04420_train.csv', path_new=r'dataset04/DS04430_train.csv')
    # PreProcess.select_feature(path=r'dataset04/DS04410s_test.csv', path_new=r'dataset04/DS04420s_test.csv')
    # PreProcess.select_feature_for_testing_set(path_training_set=r'dataset04/DS04430_train.csv',
    #                                           path=r'dataset04/DS04420_test.csv',
    #                                           path_new=r'dataset04/DS04430_test.csv')
    # PreProcess.resampling(path=r'dataset04/DS04420s_train.csv', path_new=r'dataset04/DS04430s_train.csv')

    # DS-0.4.5
    # PreProcess.remove_context_vector(path=r'dataset03/DS03014.csv', path_new=r'dataset03/DS03500.csv')
    # PreProcess.resampling(path=r'dataset03/DS03500.csv', path_new=r'dataset03/DS03501.csv')

    # DS-0.4.6
    # PreProcess.remove_context_vector_and_lexicon_metrics(path=r'dataset03/DS03014.csv', path_new=r'dataset03/DS03600.csv')
    # PreProcess.resampling(path=r'dataset03/DS03600.csv', path_new=r'dataset03/DS03601.csv')

    # DS-0.4.7
    # PreProcess.remove_systematically_generated_metrics(path=r'dataset03/DS03014.csv',
    #                                                      path_new=r'dataset03/DS03700.csv')
    # PreProcess.resampling(path=r'dataset03/DS03700.csv', path_new=r'dataset03/DS03701.csv')

    # -------------------------------------------
    # PreProcess.extract_coverage_before_and_after_refactoring('refactors/mango_statistics_before.csv',
    #                                                          'refactors/mango_statistics_after.csv')


def main():
    # Main operations
    print('STEP 1: Create udbs')
    # PreProcess.create_understand_database_from_project()
    print('-' * 75)

    print('STEP 2: Create csvs')
    # PreProcess.extract_metrics_and_coverage_all(udbs_path=database_path)

    # RQ4:
    # PreProcess.extract_metrics_and_coverage_all(udbs_path=r'refactored_projects',
    #                                             class_list_csv_path=r'runtime_result/evosuit160_sf110_result_html_with_project__refactored.csv',
    #                                             csvs_path=r'refactored_csvs/')
    # quit()
    # PreProcess.extract_metrics_and_coverage(dataset=dataset_path, database=db, project_name='5_templateit')
    print('-' * 75)

    print('STEP 3: Create single csv')
    # PreProcess.create_complete_dataset(separated_csvs_root=r'qc20_csvs_e1/',
    #                                    complete_csv_root=r'dataset06/', complete_csv_file=r'qc20_csvs_e1.csv')
    # RQ4:
    # PreProcess.create_complete_dataset(separated_csvs_root=r'refactored_csvs/',
    #                                    complete_csv_root=r'dataset06/', complete_csv_file=r'refactored_raw.csv')
    print('-' * 75)

    print('STEP 5.3: Clean dataset version 0.3.0')
    # process_dataset030()
    print('-' * 75)

    print('STEP 5.4: Clean dataset version 0.4.0')
    # process_dataset040()
    print('-' * 75)

    print('STEP 5.6: Clean dataset version 0.6.0')
    # DS-0.6.0
    # Start with DS060 (Correct columns names, Correct coverage values, Add metrics, Add two combines label)
    # PreProcess.remove_irrelevant_samples(csv_path=r'dataset06/DS060RawLabeled.csv', csv_new_path=r'dataset06/DS06010.csv')
    # PreProcess.remove_irrelevant_samples(csv_path=r'dataset06/qc20_csvs_e1.csv',
    #                                      csv_new_path=r'dataset06/qc20_csvs_e1_10.csv')
    # PreProcess.remove_zero_variance_column(path=r'dataset06/DS06010WL.csv', path_new=r'dataset06/DS06011WL.csv')

    # PreProcess.mitigate_imbalanced(path=r'dataset06/DS06010.csv', path_new=r'dataset06/DS06011.csv')
    # PreProcess.remove_dataclasses(csv_path=r'dataset06/DS06010.csv', csv_new_path=r'dataset06/DS06011.csv')

    # PreProcess.remove_outliers2(path=r'dataset06/DS06011.csv', path_new=r'dataset06/DS06012.csv')   # !DS base
    # PreProcess.remove_outliers_with_lof(path=r'dataset06/DS06011.csv', path_new=r'dataset06/DS06012.csv')  # !DS base

    # PreProcess.discretize(path=r'dataset06/DS06012B.csv', path_new=r'dataset06/DS06013C.csv')
    # PreProcess.remove_high_coverage_classes_samples(csv_path=r'dataset06/DS06012.csv', csv_new_path=r'dataset06/DS06013.csv')
    # PreProcess.discretize_q(path=r'dataset06/DS06012.csv', path_new=r'dataset06/DS06013C.csv')
    # PreProcess.label_with_line_and_branch(path=r'dataset06/DS06013C.csv', path_new=r'dataset06/DS06013D.csv')

    # PreProcess.split_dataset_base(path=r'dataset06/DS06013B.csv', path_new=r'dataset06/DS06013B_')
    # PreProcess.split_dataset_for_regression(path=r'dataset06/DS06013B.csv', path_new=r'dataset06/DS06013BR_')

    # DS-0.6.1
    # PreProcess.resampling(path=r'dataset05/DS05041_train.csv', path_new=r'dataset05/DS05111_train.csv')
    # PreProcess.resampling(path=r'dataset04/DS04050.csv', path_new=r'dataset04/DS04120.csv')

    # DS-0.6.2
    # PreProcess.normalize_features(path=r'dataset06/DS06012_train2SSL.csv',
    #                               new_path=r'dataset06/DS06012_train2SSLNR.csv')
    # PreProcess.normalize_features(path=r'dataset06/DS06012_test2.csv',
    #                               new_path=r'dataset06/DS06012_test2NR.csv')
    # PreProcess.resampling(path=r'dataset03/DS03200.csv', path_new=r'dataset03/DS03201.csv')
    # PreProcess.extract_feature(path='dataset05/DS05423_trainX5.csv', path_new=r'dataset05/DS05423_trainX6.csv', number_of_features=10)

    # DS-0.6.3
    # PreProcess.select_feature(path=r'dataset05/DS05041_train.csv', path_new=r'dataset05/DS05311_train.csv')
    # PreProcess.select_feature_for_testing_set(path_training_set=r'dataset06/DS06012_train2.csv',
    #                                           path_testing_set=r'dataset06/DS06012_test.csv',
    #                                           path_testing_set_new=r'dataset06/DS06012_test2.csv')
    # PreProcess.select_feature_for_testing_set(path_training_set=r'dataset06/DS06012_train2.csv',
    #                                           path_testing_set=r'dataset06/qc20_csvs_e1_11.csv',
    #                                           path_testing_set_new=r'dataset06/qc20_csvs_e1_12.csv')
    # PreProcess.resampling(path=r'dataset03/DS03300.csv', path_new=r'dataset03/DS03301.csv')

    # DS-0.6.4
    # PreProcess.normalize_features(path=r'dataset05/DS05023_trainX4.csv', new_path=r'dataset05/DS05423_train.csv')
    # PreProcess.normalize_features(path=r'dataset05/DS05423_test.csv', new_path=r'dataset05/DS05423_testX5.csv')

    # PreProcess.select_feature(path=r'dataset05/DS05410_train.csv', path_new=r'dataset05/DS05411_train.csv')

    # PreProcess.select_feature(path=r'dataset04/DS04410s_test.csv', path_new=r'dataset04/DS04420s_test.csv')
    # PreProcess.select_feature_for_testing_set(path_training_set=r'dataset05/DS05411_train.csv',
    #                                           path=r'dataset05/DS05410_test.csv',
    #                                           path_new=r'dataset05/DS05411_test.csv')

    # PreProcess.resampling(path=r'dataset05/DS05411_train.csv', path_new=r'dataset05/DS05412_train.csv')

    # DS-0.6.5/new: 3
    # PreProcess.remove_context_vector(path=r'dataset06/DS06013.csv', path_new=r'dataset06/DS06310.csv')
    # PreProcess.resampling(path=r'dataset03/DS03500.csv', path_new=r'dataset03/DS03501.csv')

    # DS-0.6.6/new: 4
    # PreProcess.remove_context_vector_and_lexicon_metrics(path=r'dataset06/DS06013.csv', path_new=r'dataset06/DS06410.csv')
    # PreProcess.resampling(path=r'dataset03/DS03600.csv', path_new=r'dataset03/DS03601.csv')

    # DS-0.6.7/new: 5
    # PreProcess.remove_systematically_generated_metrics(path=r'dataset06/DS06013.csv', path_new=r'dataset06/DS06510.csv')
    # PreProcess.resampling(path=r'dataset03/DS03700.csv', path_new=r'dataset03/DS03701.csv')

    # -------------------------------------------
    # PreProcess.extract_coverage_before_and_after_refactoring('refactors/mango_statistics_before.csv',
    #                                                          'refactors/mango_statistics_after.csv')

    # PreProcess.prepared_model_input_for_inference(r'dataset06/refactored_raw.csv', r'dataset06/refactored01010.csv')

    PreProcess.create_complete_dataset(
        separated_csvs_root=r'sf110_csvs_without_test/',
        complete_csv_root=r'dataset06/',
        complete_csv_file=r'DS060Raw.csv',
    )


if __name__ == '__main__':
    main()
