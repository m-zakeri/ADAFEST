# ADAFEST datasets


The ADAFEST datasets contains all experimental data used in our machine learning pipeline at ADAFEST project.
Our testability prediction dataset consists of several CSV files which differ in preprocessing steps used to generate them. Each row denotes metrics for a Java class. Each column is a source code metrics or test metrics obtained by running EvoSuite on the corresponding class under test. The first column is a long name (package_name.class_name) of a Java class. More information will be available in [ADAFEST publications](./publications.md).
All data available on Zenodo:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4650228.svg)](https://doi.org/10.5281/zenodo.4650228)



The current version of the testability prediction dataset (dataset06—version 0.6.x) contains the following files:

1. `DS060Raw.csv`: Contains only source code metrics for 19,720 Java classes. The last column indicates the number of Java classes in the enclosing file of the presented Java class. Actually, SF110 contains more than 23K Java classes. We removed small projects and projects that most of their classes are data classes used as database models.

2. `DS060RawLabeled.csv`: The same `DS060Raw.csv` with ten attached columns containing dynamically computed metrics obtained by running EvoSuite test data generation tools. The last four columns are combinatory metrics computed based on the primary metrics given by EvoSuite.  The most useful metrics are statement coverage, branch coverage, and the number of generated tests. More details of EvoSuite configuration and runtime metrics are available in ADAFEST relevant papers.

3. `DS06010.csv` and `DS06011.csv`: This file contains 18,324 Java classes. Irrelevant samples in `DS060RawLabeled.csv` (i.e., simple classes, data class, files with more than one class, classed with zero number of test cases) have been removed in this file.

4. `DS06012.csv`: Class with outlier metrics have been deleted from `DS06011.csv`, and this file contains 16,165 Java classes.

5. `DS06012_outliers_only.csv`: This file contains Java classes detected as an outlier by the local outlier factor (LOF) algorithm.

6. `DS06310.csv`: Package metrics (used as context vector in our testability prediction approach) have been removed from `DS06012.csv` in this file.

7. `DS06410.csv`: Package metrics (used as context vector in our testability prediction approach) and lexical metrics have been removed from DS06012.csv in this file.

8. `DS06510.csv`: Sub-metrics (systematically generated metrics) have been removed from DS06012.csv in this file.


